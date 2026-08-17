import asyncio
import json
import time
import aiohttp
from decimal import Decimal
from typing import Optional, Dict, Literal


from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
from web3.contract import Contract
from loguru import logger

from ..core.config import DexConfig, NetworkConfig, SecretsConfig, TokenDetails
from ..core.types import MarketPair, DexQuote, DexSwapParams, DexTxReceipt
from .dex_base import DexClient

ZERO_DEC = Decimal("0")

# --- ABI helpers ---
def _load_abi(path: str) -> Dict:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"ABI file not found: {path}")
        raise

class UniV3DexClient(DexClient):
    def __init__(self, dex_config: DexConfig, net_config: NetworkConfig, secrets: SecretsConfig, tokens_config: Dict[str, Dict[str, 'TokenDetails']]):
        self.dex_config = dex_config
        self.net_config = net_config
        self.secrets = secrets
        self.tokens_config = tokens_config  # Directly use the passed dictionary
        self.user_address = Web3.to_checksum_address(Web3().eth.account.from_key(secrets.dex_wallet_private_key.get_secret_value()).address)
        
        self.w3_instances: Dict[str, Web3] = {}
        for chain, rpc_url in net_config.rpc_urls.items():
            if rpc_url:
                w3 = Web3(Web3.HTTPProvider(rpc_url))
                if chain in ["bsc", "polygon", "base"]:
                    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
                self.w3_instances[chain] = w3

        self.quoter_abi = _load_abi("ABI/quoter.json")
        self.router_abi = _load_abi("ABI/router.json")
        self.erc20_abi = _load_abi("ABI/erc20.json")
        try:
            self.factory_abi = _load_abi("ABI/factory.json")
        except Exception:
            self.factory_abi = None
            logger.debug("Uniswap V3 factory ABI not loaded; some features will be unavailable.")
        self._factory_contracts: Dict[str, 'Contract'] = {}
        logger.info(f"UniV3DexClient initialised, address: {self.user_address}, supported chains: {list(self.w3_instances.keys())}")

    def _to_atomic(self, amount: Decimal, decimals: int) -> int:
        return int(amount * (10**decimals))

    def _from_atomic(self, amount: Decimal, decimals: int) -> Decimal:
        return amount / (10**decimals)

    def _get_w3(self, chain: str) -> Web3:
        if chain not in self.w3_instances:
            raise ValueError(f"Unsupported chain, or its RPC URL is not configured: {chain}")
        return self.w3_instances[chain]

    def _get_factory_contract(self, chain: str):
        if not self.factory_abi:
            raise ValueError("Factory ABI not loaded; cannot check whether a pool exists.")
        if chain not in self.dex_config.uniswap_v3:
            raise ValueError(f"No DEX contract details configured for chain {chain}.")

        contracts = self.dex_config.uniswap_v3[chain]
        if not getattr(contracts, 'factory', None):
            raise ValueError(f"Chain {chain} has no Uniswap V3 factory address configured.")

        if chain not in self._factory_contracts:
            factory_address = Web3.to_checksum_address(contracts.factory)
            factory_contract = self._get_w3(chain).eth.contract(address=factory_address, abi=self.factory_abi)
            self._factory_contracts[chain] = factory_contract
        return self._factory_contracts[chain]

    def _get_token_details(self, symbol: str, chain: str) -> 'TokenDetails':
        if symbol not in self.tokens_config or chain not in self.tokens_config[symbol]:
            raise ValueError(f"No address configured for token {symbol} on chain {chain}.")
        return self.tokens_config[symbol][chain]

    async def get_pool_address(self, base_symbol: str, quote_symbol: str, chain: str, fee: int) -> Optional[str]:
        """Return the pool address for the given token pair and fee tier if it exists."""
        try:
            factory = self._get_factory_contract(chain)
        except ValueError as exc:
            logger.debug(f"Failed to access the factory contract: {exc}")
            return None

        try:
            base = self._get_token_details(base_symbol, chain)
            quote = self._get_token_details(quote_symbol, chain)
        except ValueError as exc:
            logger.debug(f"Failed to obtain token details: {exc}")
            return None

        token_a = Web3.to_checksum_address(base.address)
        token_b = Web3.to_checksum_address(quote.address)

        def _call() -> str:
            return factory.functions.getPool(token_a, token_b, int(fee)).call()

        try:
            pool_address = await asyncio.to_thread(_call)
        except Exception as exc:
            logger.warning(f"Uniswap pool lookup failed ({base_symbol}/{quote_symbol} {chain} fee={fee}): {exc}")
            return None

        if pool_address and int(pool_address, 16) != 0:
            return Web3.to_checksum_address(pool_address)
        return None

    async def get_balance(self, asset: str, chain: str) -> Decimal:
        """Return the wallet balance of a token on a specific chain."""
        w3 = self._get_w3(chain)
        try:
            # native token, e.g. ETH on mainnet
            if asset == self.net_config.native_token.get(chain):
                balance_wei = await asyncio.to_thread(w3.eth.get_balance, self.user_address)
                balance = Decimal(balance_wei) / Decimal(10**18)
                logger.debug(f"Native token {asset} balance on {chain}: {balance}")
                return balance

            # ERC20 token
            token_details = self._get_token_details(asset, chain)
            token_address = Web3.to_checksum_address(token_details.address)
            token_contract = w3.eth.contract(address=token_address, abi=self.erc20_abi)
            
            balance_wei = await asyncio.to_thread(
                token_contract.functions.balanceOf(self.user_address).call
            )
            
            balance = Decimal(balance_wei) / (Decimal(10) ** token_details.decimals)
            logger.debug(f"{asset} balance on {chain}: {balance}")
            return balance

        except ValueError as e:
            logger.error(f"Balance lookup failed: {e}")
            return Decimal("-1")
        except Exception as e:
            logger.error(f"Unknown error while fetching the {asset} balance on {chain}: {e}")
            return Decimal("-1")

    async def get_quote(
        self, 
        pair: MarketPair, 
        size: Decimal, 
        side: Literal["buy", "sell"],
        estimate_gas: bool = False
    ) -> Optional[DexQuote]:
        """Fetch a trade quote from Uniswap V3."""
        try:
            w3 = self._get_w3(pair.dex_chain)
            
            # Prioritize address/decimals from the pair object itself.
            if pair.base_address and pair.quote_address:
                base_token = TokenDetails(address=pair.base_address, decimals=pair.base_decimals)
                quote_token = TokenDetails(address=pair.quote_address, decimals=pair.quote_decimals)
            else:
                # Fallback to looking up from the config.
                base_token = self._get_token_details(pair.base, pair.dex_chain)
                quote_token = self._get_token_details(pair.quote_dex, pair.dex_chain)

        except ValueError as e:
            logger.warning(f"Failed to fetch the DEX quote ({pair.cex_symbol} on {pair.dex_chain}): {e}")
            return None

        try:
            # === derive amountIn and tokenIn/tokenOut ===
            if side == "sell":
                # base in, quote out
                amount_in_atomic = self._to_atomic(size, base_token.decimals)
                token_in = Web3.to_checksum_address(base_token.address)
                token_out = Web3.to_checksum_address(quote_token.address)
            else:
                # side == "buy": quote in, spending quote to receive base
                amount_in_atomic = self._to_atomic(size, quote_token.decimals)
                token_in = Web3.to_checksum_address(quote_token.address)
                token_out = Web3.to_checksum_address(base_token.address)

            fee = int(pair.dex_pool_fee)

            # === QuoterV2 ===
            quoter_address = Web3.to_checksum_address(self.dex_config.uniswap_v3[pair.dex_chain].quoter_v2)
            quoter = w3.eth.contract(address=quoter_address, abi=self.quoter_abi)

            # QuoterV2's quoteExactInputSingle takes a single struct
            # field order: (tokenIn, tokenOut, amountIn, fee, sqrtPriceLimitX96)
            params = (token_in, token_out, int(amount_in_atomic), fee, 0)

            # .call() is synchronous; run it in a thread so the loop is not blocked
            raw = await asyncio.to_thread(lambda: quoter.functions.quoteExactInputSingle(params).call())

            # different ABIs return either (amountOut, sqrtPriceX96After, ticks, gas) or just amountOut
            amount_out_atomic = int(raw[0]) if isinstance(raw, (list, tuple)) else int(raw)

            # === convert to a price, accounting for token decimals ===
            if side == "sell":
                # base in, quote out => price = quote_out / base_in
                amount_in_base = Decimal(amount_in_atomic) / (Decimal(10) ** base_token.decimals)
                amount_out_quote = Decimal(amount_out_atomic) / (Decimal(10) ** quote_token.decimals)
                price = (amount_out_quote / amount_in_base) if amount_in_base > 0 else ZERO_DEC
            else:
                # quote in, base out => price = quote_in / base_out
                amount_in_quote = Decimal(amount_in_atomic) / (Decimal(10) ** quote_token.decimals)
                amount_out_base = Decimal(amount_out_atomic) / (Decimal(10) ** base_token.decimals)
                price = (amount_in_quote / amount_out_base) if amount_out_base > 0 else ZERO_DEC

            gas_cost_quote = ZERO_DEC

            if estimate_gas:
                gas_estimate = 200_000  # rough estimate
                gas_price_wei = await asyncio.to_thread(lambda: w3.eth.gas_price)
                # the cost is denominated in the native token (e.g. ETH); expressed here in USD
                native_token_price_usd = await self._get_native_token_price_usd(pair.dex_chain)
                gas_cost_native = w3.from_wei(gas_estimate * gas_price_wei, 'ether')
                gas_cost_quote = Decimal(str(gas_cost_native)) * Decimal(str(native_token_price_usd))

            return DexQuote(price=price, gas_cost_quote=gas_cost_quote)

        except Exception as e:
            logger.warning(f"Error fetching a quote from QuoterV2 ({pair.cex_symbol} on {pair.dex_chain}): {e}")
            return None

    async def _get_native_token_price_usd(self, chain: str) -> Decimal:
        try:
            if chain in ['ethereum', 'arbitrum', 'base']:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd",
                        timeout=8
                    ) as response:
                        data = await response.json()
                        return Decimal(str(data['ethereum']['usd']))
        except Exception as _:
            pass
        return Decimal("3000")

    async def _approve_token(self, w3: Web3, token_address: str, router_address: str, required_amount: int):
        token_contract = w3.eth.contract(address=token_address, abi=self.erc20_abi)
        
        allowance = await asyncio.to_thread(
            token_contract.functions.allowance(self.user_address, router_address).call
        )
        if allowance >= required_amount:
            logger.debug(f"Token {token_address} already has sufficient allowance.")
            return True

        logger.info(f"Approving token {token_address} for router {router_address}...")
        amount_to_approve = 2**256 - 1 # Approve max
        nonce = await asyncio.to_thread(w3.eth.get_transaction_count, self.user_address)
        
        tx_params = {
            'from': self.user_address,
            'nonce': nonce,
            'gasPrice': await asyncio.to_thread(lambda: w3.eth.gas_price),
        }
        
        approve_tx = await asyncio.to_thread(
            token_contract.functions.approve(router_address, amount_to_approve).build_transaction, tx_params
        )
        
        signed_tx = w3.eth.account.sign_transaction(approve_tx, self.secrets.dex_wallet_private_key.get_secret_value())
        tx_hash = await asyncio.to_thread(w3.eth.send_raw_transaction, signed_tx.rawTransaction)
        logger.info(f"Approval transaction sent: {tx_hash.hex()}")
        
        receipt = await asyncio.to_thread(w3.eth.wait_for_transaction_receipt, tx_hash, timeout=self.net_config.max_pending_seconds)
        if receipt['status'] == 1:
            logger.success(f"Token {token_address} approved successfully.")
            return True
        else:
            logger.error(f"Token {token_address} approval failed. Receipt: {receipt}")
            return False

    async def execute_swap(self, params: DexSwapParams) -> DexTxReceipt:
        w3 = self._get_w3(params.chain)
        router_address = Web3.to_checksum_address(self.dex_config.uniswap_v3[params.chain].router)
        token_in_addr = Web3.to_checksum_address(params.token_in_address)
        token_out_addr = Web3.to_checksum_address(params.token_out_address)
        amount_in_wei = int(params.amount_in * (10**params.token_in_decimals))

        await self._approve_token(w3, token_in_addr, router_address, amount_in_wei)

        logger.info(f"Preparing DEX swap on {params.chain}: {params.amount_in} of token {params.token_in_address}")
        router_contract = w3.eth.contract(address=router_address, abi=self.router_abi)
        nonce = await asyncio.to_thread(w3.eth.get_transaction_count, self.user_address)

        amount_out_minimum = 0 # MUST be derived from the quote and max_slippage_bps before production use

        swap_params_struct = {
            'tokenIn': token_in_addr,
            'tokenOut': token_out_addr,
            'fee': params.fee,
            'recipient': self.user_address,
            'deadline': int(time.time()) + 600,
            'amountIn': amount_in_wei,
            'amountOutMinimum': amount_out_minimum,
            'sqrtPriceLimitX96': 0
        }

        tx_params = {
            'from': self.user_address,
            'nonce': nonce,
            'gasPrice': await asyncio.to_thread(lambda: w3.eth.gas_price),
        }
        tx_params['gas'] = await asyncio.to_thread(router_contract.functions.exactInputSingle(swap_params_struct).estimate_gas, {'from': self.user_address, 'value': 0})

        try:
            swap_tx = await asyncio.to_thread(
                router_contract.functions.exactInputSingle(swap_params_struct).build_transaction, tx_params
            )
            signed_tx = w3.eth.account.sign_transaction(swap_tx, self.secrets.dex_wallet_private_key.get_secret_value())
            tx_hash = await asyncio.to_thread(w3.eth.send_raw_transaction, signed_tx.rawTransaction)
            logger.info(f"DEX swap transaction sent: {tx_hash.hex()}")

            receipt = await asyncio.to_thread(w3.eth.wait_for_transaction_receipt, tx_hash, timeout=self.net_config.max_pending_seconds)
            
            if receipt['status'] == 1:
                logger.success(f"DEX swap succeeded. Tx: {tx_hash.hex()}")
                # TODO: parse the actual fill price and size from the transaction event logs
                return DexTxReceipt(
                    tx_hash=tx_hash.hex(), status=1, block_number=receipt['blockNumber'],
                    gas_used=receipt['gasUsed'], effective_gas_price=receipt['effectiveGasPrice'],
                    avg_fill_price=Decimal(0), filled_size=Decimal(0) # Placeholder
                )
            else:
                logger.error(f"DEX swap failed. Tx: {tx_hash.hex()}, receipt: {receipt}")
                raise Exception("DEX transaction reverted")

        except Exception as e:
            logger.error(f"DEX swap execution failed: {e}", exc_info=True)
            raise
