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

from ..core import clock
from ..core.config import DexConfig, NetworkConfig, SecretsConfig, TokenDetails
from .price_oracle import NativePriceOracle
from ..core.types import MarketPair, DexQuote, DexSwapParams, DexTxReceipt
from .dex_base import DexClient
from .errors import ReadOnlyWalletError, RpcError, classify_rpc_failure
from .rpc_limit import RpcLimiter

TEN_THOUSAND = Decimal("10000")


def min_amount_out_wei(
    expected_out: Decimal, slippage_bps: int, token_out_decimals: int
) -> int:
    """The router's `amountOutMinimum`, in the token's integer units.

    This is the only protection against a sandwich attack. The previous code
    passed 0 under a comment saying it MUST be derived first: with a zero floor
    the router accepts any output at all, so an attacker moves the pool, our swap
    executes at whatever price results, and the attacker closes. The loss is
    bounded by the pool's liquidity rather than by the trade size.

    Rounds DOWN. Rounding up would set a floor above what the quote promised, so a
    swap that filled exactly as quoted would revert -- a failure that presents as
    a market problem while actually being an arithmetic one.

    `slippage_bps` is a tolerance, not a cost: it never enters the trade
    economics. It only decides how far from the quote a fill may land before the
    router refuses it.
    """
    if expected_out <= 0:
        raise ValueError(
            f"expected_out must be positive, got {expected_out}; there is no "
            f"meaningful floor below a non-positive expectation"
        )
    if slippage_bps < 0:
        raise ValueError(
            f"slippage_bps must not be negative, got {slippage_bps}. A negative "
            f"tolerance demands more than the quote and reverts every swap."
        )
    if slippage_bps >= 10_000:
        raise ValueError(
            f"slippage_bps must be below 10000 (100%), got {slippage_bps}. A "
            f"whole-turn tolerance makes the floor zero, which is exactly the "
            f"unprotected case this function exists to prevent."
        )
    if not 0 <= token_out_decimals <= 36:
        raise ValueError(
            f"token_out_decimals is {token_out_decimals}, outside 0..36"
        )

    floor = expected_out * (TEN_THOUSAND - Decimal(slippage_bps)) / TEN_THOUSAND
    # int() truncates toward zero, which is the rounding-down this needs.
    raw = int(floor * (Decimal(10) ** token_out_decimals))

    if raw <= 0:
        # Found by a property test over sizes and decimals: 1 raw unit of
        # expected output with a 1 bps tolerance rounds to a floor of zero, which
        # is the unprotected case. Clamping to 1 would convert "no protection"
        # into a fiction, so this refuses instead. Such a trade is dust -- gas
        # alone exceeds its entire output by orders of magnitude -- and the
        # sizing checks upstream should never produce it.
        raise ValueError(
            f"a floor of {floor} token-out units rounds to zero at "
            f"{token_out_decimals} decimals, so this swap cannot be protected. "
            f"Expected output {expected_out} is at the token's resolution limit; "
            f"refusing to send an unprotected swap."
        )
    return raw

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
    def __init__(self, dex_config: DexConfig, net_config: NetworkConfig, secrets: SecretsConfig, tokens_config: Dict[str, Dict[str, 'TokenDetails']],
        rpc_limiter: Optional[RpcLimiter] = None,
    ):
        self.dex_config = dex_config
        self.net_config = net_config
        self.secrets = secrets
        self.tokens_config = tokens_config  # Directly use the passed dictionary
        # None when no signing key was supplied. A read-only client is a real
        # configuration -- the recorder, the survey and the backtest all quote
        # pools without ever holding a wallet -- so construction must not require
        # one. Every signing path guards on it explicitly below.
        if secrets.dex_wallet_private_key is None:
            self.user_address = None
        else:
            self.user_address = Web3.to_checksum_address(
                Web3().eth.account.from_key(
                    secrets.dex_wallet_private_key.get_secret_value()
                ).address
            )
        
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
        # Native-token USD price, cached with an explicit staleness contract.
        # Fails closed: if gas cannot be priced, the quote is declined.
        # Paced per chain. The exchange side has had a weight governor for a
        # while; the chain side had nothing, and a universe survey against
        # public endpoints drew sustained 429s -- which were reported upward as
        # "no pool" until the attribution fix.
        self._rpc_limiter = rpc_limiter or RpcLimiter(
            requests_per_second=net_config.rpc_requests_per_second,
            max_concurrency=net_config.rpc_max_concurrency,
            per_chain_requests_per_second=net_config.rpc_requests_per_second_by_chain,
        )
        logger.info(self._rpc_limiter.describe())
        self.price_oracle = NativePriceOracle(
            ttl_seconds=dex_config.native_price_ttl_seconds,
            stale_grace_seconds=dex_config.native_price_stale_grace_seconds,
        )
        logger.info(f"UniV3DexClient initialised, address: {self.user_address}, supported chains: {list(self.w3_instances.keys())}")


    def _sign_transaction(self, transaction: dict, chain: Optional[str] = None):
        """Sign a transaction, or refuse loudly if this client is read-only.

        Centralised deliberately. Each signing site used to reach into
        `secrets.dex_wallet_private_key` itself, so a read-only client would have
        raised AttributeError from inside web3 -- indistinguishable from a bug, at
        the worst possible moment. One guarded helper means a new signing path
        inherits the check instead of having to remember it.
        """
        if self.secrets.dex_wallet_private_key is None:
            raise ReadOnlyWalletError(
                "this client is read-only: no DEX_WALLET_PRIVATE_KEY was supplied, "
                "so it cannot sign transactions. Research, recording and backtest "
                "runs are read-only by design; if this is a live run, the wallet "
                "key is missing from the environment."
            )
        w3 = self._w3(chain) if chain else Web3()
        return w3.eth.account.sign_transaction(
            transaction, self.secrets.dex_wallet_private_key.get_secret_value()
        )

    def _to_atomic(self, amount: Decimal, decimals: int) -> int:
        return int(amount * (10**decimals))

    def _from_atomic(self, amount: Decimal, decimals: int) -> Decimal:
        return amount / (10**decimals)

    async def _rpc(self, chain: str, fn, *args, **kwargs):
        """Every chain call goes through here, paced per chain.

        A single chokepoint rather than a limiter at each of sixteen call sites:
        the accounting cannot then be bypassed by adding a seventeenth, which is
        exactly how the exchange side went unmetered for so long. A test asserts
        via the AST that no `asyncio.to_thread` in this module skips it.

        Long polls are deliberately NOT routed through here -- see
        `_rpc_unpaced`. Holding a concurrency slot for the length of a receipt wait
        would let one pending transaction starve every quote.
        """
        async with self._rpc_limiter.acquire(chain):
            return await asyncio.to_thread(fn, *args, **kwargs)

    @staticmethod
    async def _rpc_unpaced(fn, *args, **kwargs):
        """A chain call that must not hold a limiter slot.

        Only for waits whose duration is set by the chain rather than by us: a
        receipt poll can take a minute, and web3 does its own internal polling, so
        occupying a slot for that long would starve the hot loop while protecting
        nothing.
        """
        return await asyncio.to_thread(fn, *args, **kwargs)

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

    async def get_pool_address_by_tokens(
        self, token_a_address: str, token_b_address: str, chain: str, fee: int
    ) -> Optional[str]:
        """Pool address for two token ADDRESSES, or None.

        The factory only ever needed addresses; requiring symbols meant a pool
        could not be found for a token absent from `tokens.yaml`, which made
        surveying a universe wider than the configured one impossible. Ordering
        does not matter -- the factory sorts internally.
        """
        try:
            factory = self._get_factory_contract(chain)
        except ValueError as exc:
            logger.debug(f"Failed to access the factory contract: {exc}")
            return None

        try:
            token_a = Web3.to_checksum_address(token_a_address)
            token_b = Web3.to_checksum_address(token_b_address)
        except Exception as exc:
            logger.debug(f"Bad token address in a pool lookup: {exc}")
            return None

        def _call() -> str:
            return factory.functions.getPool(token_a, token_b, int(fee)).call()

        try:
            pool_address = await self._rpc(chain, _call)
        except Exception as exc:
            if classify_rpc_failure(exc):
                # Surfaced rather than swallowed: a throttled lookup reported as
                # "no pool" would permanently exclude a pool that does exist.
                raise RpcError(f"{chain}: {type(exc).__name__}: {exc}") from exc
            logger.debug(
                f"Uniswap pool lookup failed ({token_a}/{token_b} {chain} "
                f"fee={fee}): {exc}"
            )
            return None

        if pool_address and int(pool_address, 16) != 0:
            return Web3.to_checksum_address(pool_address)
        return None

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
            pool_address = await self._rpc(chain, _call)
        except Exception as exc:
            logger.warning(f"Uniswap pool lookup failed ({base_symbol}/{quote_symbol} {chain} fee={fee}): {exc}")
            return None

        if pool_address and int(pool_address, 16) != 0:
            return Web3.to_checksum_address(pool_address)
        return None

    async def quote_exact_input_single_raw(
        self, *, chain: str, token_in: str, token_out: str, fee: int,
        amount_in: int, block_number: Optional[int] = None,
    ) -> int:
        """QuoterV2's raw integer output, at a specific block.

        Kept as an ORACLE, not as the hot path. The local simulator in
        `univ3_math` is what prices a trade; this is what the simulator is checked
        against, and pinning the block is what makes that comparison meaningful --
        otherwise the two answers can differ simply because the pool moved between
        them, and a real disagreement would be indistinguishable from a race.
        """
        w3 = self._get_w3(chain)
        quoter_address = Web3.to_checksum_address(
            self.dex_config.uniswap_v3[chain].quoter_v2
        )
        quoter = w3.eth.contract(address=quoter_address, abi=self.quoter_abi)
        params = (
            Web3.to_checksum_address(token_in),
            Web3.to_checksum_address(token_out),
            int(amount_in),
            int(fee),
            0,
        )

        def _call():
            call = quoter.functions.quoteExactInputSingle(params)
            if block_number is None:
                return call.call()
            return call.call(block_identifier=block_number)

        raw = await self._rpc(chain, _call)
        return int(raw[0]) if isinstance(raw, (list, tuple)) else int(raw)

    async def get_balance(self, asset: str, chain: str) -> Decimal:
        """Return the wallet balance of a token on a specific chain."""
        w3 = self._get_w3(chain)
        try:
            # native token, e.g. ETH on mainnet
            if asset == self.net_config.native_token.get(chain):
                balance_wei = await self._rpc(chain, w3.eth.get_balance, self.user_address)
                balance = Decimal(balance_wei) / Decimal(10**18)
                logger.debug(f"Native token {asset} balance on {chain}: {balance}")
                return balance

            # ERC20 token
            token_details = self._get_token_details(asset, chain)
            token_address = Web3.to_checksum_address(token_details.address)
            token_contract = w3.eth.contract(address=token_address, abi=self.erc20_abi)
            
            balance_wei = await self._rpc(
                chain, token_contract.functions.balanceOf(self.user_address).call
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
            raw = await self._rpc(
                pair.dex_chain,
                lambda: quoter.functions.quoteExactInputSingle(params).call(),
            )

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
                gas_price_wei = await self._rpc(
                    pair.dex_chain, lambda: w3.eth.gas_price
                )
                priced = await self._gas_cost_in_quote(
                    oracle=self.price_oracle,
                    chain=pair.dex_chain,
                    gas_units=self.dex_config.swap_gas_estimate_units,
                    gas_price_wei=int(gas_price_wei),
                )
                if priced is None:
                    logger.warning(
                        f"Declining to quote {pair.cex_symbol} on {pair.dex_chain}: "
                        f"gas cannot be priced, so the economics are not trustworthy."
                    )
                    return None
                gas_cost_quote = priced

            return DexQuote(price=price, gas_cost_quote=gas_cost_quote)

        except Exception as e:
            # A node that did not answer is a different fact from a pool with no
            # liquidity, and collapsing both into None made them the same row in
            # the audit trail. Under RPC pressure the bot would appear to be
            # watching an empty market.
            if classify_rpc_failure(e):
                logger.warning(
                    f"RPC failure quoting {pair.cex_symbol} on {pair.dex_chain}: "
                    f"{type(e).__name__}: {e}"
                )
                raise RpcError(
                    f"{pair.dex_chain}: {type(e).__name__}: {e}"
                ) from e
            logger.warning(f"Error fetching a quote from QuoterV2 ({pair.cex_symbol} on {pair.dex_chain}): {e}")
            return None

    @staticmethod
    async def _gas_cost_in_quote(
        *,
        oracle: NativePriceOracle,
        chain: str,
        gas_units: int,
        gas_price_wei: int,
    ) -> Optional[Decimal]:
        """Convert a gas cost into the quote currency, or None if unpriceable.

        Returning None is the important behaviour. A guessed native price
        silently rescales every gas cost and therefore silently shifts every
        profitability decision, with nothing raised and nothing logged.
        """
        native_price = await oracle.get_usd_price(chain)
        if native_price is None:
            return None
        gas_cost_native = Decimal(gas_units) * Decimal(gas_price_wei) / Decimal(10 ** 18)
        return gas_cost_native * native_price

    async def _fee_params(self, w3: Web3) -> Dict[str, int]:
        """EIP-1559 fee fields, from config values that nothing previously read.

        `network.priority_fee_gwei` and `network.max_fee_gwei` existed in config
        and were used nowhere: every transaction went out with the legacy
        `gasPrice` field. A legacy transaction is still accepted post-London, but
        its fee cannot be adjusted afterwards, so one submitted into a rising base
        fee simply sits until it is dropped -- with an arbitrage leg already
        executed on the other venue.

        The configured max fee is a ceiling on what a trade may pay. It is
        deliberately not derived from the current base fee: an automatic multiple
        would let a fee spike spend an unbounded amount of the edge.
        """
        priority_wei = int(
            Decimal(str(self.net_config.priority_fee_gwei)) * Decimal(10**9)
        )
        max_wei = int(Decimal(str(self.net_config.max_fee_gwei)) * Decimal(10**9))
        if priority_wei > max_wei:
            raise ValueError(
                f"network.priority_fee_gwei ({self.net_config.priority_fee_gwei}) "
                f"exceeds network.max_fee_gwei ({self.net_config.max_fee_gwei}); "
                f"the transaction would be rejected as malformed."
            )
        return {
            'maxPriorityFeePerGas': priority_wei,
            'maxFeePerGas': max_wei,
        }

    async def _approve_token(
        self, w3: Web3, chain: str, token_address: str, router_address: str,
        required_amount: int,
    ):
        token_contract = w3.eth.contract(address=token_address, abi=self.erc20_abi)

        allowance = await self._rpc(
            chain,
            token_contract.functions.allowance(self.user_address, router_address).call,
        )
        if allowance >= required_amount:
            logger.debug(f"Token {token_address} already has sufficient allowance.")
            return True

        logger.info(f"Approving token {token_address} for router {router_address}...")
        # Exactly what this swap spends, not an unlimited allowance. The previous
        # value was 2**256 - 1, which makes the blast radius of a compromised or
        # misconfigured router the entire balance of the token rather than one
        # trade. The cost is one approval per swap instead of one ever; at a
        # 200k-gas swap on a sub-gwei chain that is a rounding error against the
        # exposure it removes.
        amount_to_approve = required_amount
        nonce = await self._rpc(
            chain, w3.eth.get_transaction_count, self.user_address
        )

        tx_params = {
            'from': self.user_address,
            'nonce': nonce,
            **await self._fee_params(w3),
        }

        approve_tx = await self._rpc(
            chain,
            token_contract.functions.approve(router_address, amount_to_approve).build_transaction,
            tx_params,
        )
        
        signed_tx = self._sign_transaction(approve_tx)
        tx_hash = await self._rpc(
            chain, w3.eth.send_raw_transaction, signed_tx.raw_transaction
        )
        logger.info(f"Approval transaction sent: {tx_hash.hex()}")
        
        # Unpaced: a receipt wait is as long as the chain makes it, and holding
        # a limiter slot for that would starve every quote.
        receipt = await self._rpc_unpaced(
            w3.eth.wait_for_transaction_receipt, tx_hash,
            timeout=self.net_config.max_pending_seconds,
        )
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

        await self._approve_token(
            w3, params.chain, token_in_addr, router_address, amount_in_wei
        )

        logger.info(f"Preparing DEX swap on {params.chain}: {params.amount_in} of token {params.token_in_address}")
        router_contract = w3.eth.contract(address=router_address, abi=self.router_abi)
        nonce = await self._rpc(
            params.chain, w3.eth.get_transaction_count, self.user_address
        )

        # From the caller's floor, which DexSwapParams requires to be positive.
        # The router enforces it on-chain: the swap reverts rather than filling
        # at a price a sandwich attack chose.
        amount_out_minimum = min_amount_out_wei(
            params.min_amount_out, 0, params.token_out_decimals
        )

        # Exactly the seven fields ABI/router.json declares, in its order.
        #
        # This previously included a `deadline`, which the struct does not have.
        # Verified against the deployed contracts: the ABI is the SwapRouter02 ABI
        # (selector 0x04e45aaf, seven fields, no deadline) and all three configured
        # routers dispatch that selector -- so the ABI and the chain agreed, and the
        # CALLER was wrong. The failure would have been at encoding time, in this
        # process, on the first real swap.
        #
        # Consequence worth stating rather than papering over: SwapRouter02's
        # exactInputSingle cannot take a deadline at all. Deadline protection there
        # is `multicall(uint256 deadline, bytes[] data)`, which ABI/router.json does
        # not include. So `dex.swap_deadline_seconds` is UNENFORCEABLE on this path.
        # For an arbitrage swap that matters -- one landing late is a guaranteed
        # loss, not a late win -- so wrapping this call in multicall is required
        # before any real execution, and it is listed as such in the README.
        swap_params_struct = {
            'tokenIn': token_in_addr,
            'tokenOut': token_out_addr,
            'fee': params.fee,
            'recipient': self.user_address,
            'amountIn': amount_in_wei,
            'amountOutMinimum': amount_out_minimum,
            'sqrtPriceLimitX96': 0,
        }
        logger.warning(
            f"Swap has NO deadline: SwapRouter02's exactInputSingle does not accept "
            f"one, and dex.swap_deadline_seconds "
            f"({self.dex_config.swap_deadline_seconds}s) cannot be applied here. "
            f"Wrap this call in multicall(deadline, data) before trading real size."
        )
        logger.info(
            f"Swap floor: {params.min_amount_out} token-out units "
            f"({amount_out_minimum} raw), from a {params.slippage_bps} bps "
            f"tolerance. The router reverts below this."
        )

        tx_params = {
            'from': self.user_address,
            'nonce': nonce,
            **await self._fee_params(w3),
        }
        tx_params['gas'] = await self._rpc(
            params.chain,
            router_contract.functions.exactInputSingle(swap_params_struct).estimate_gas,
            {'from': self.user_address, 'value': 0},
        )

        try:
            swap_tx = await self._rpc(
                params.chain,
                router_contract.functions.exactInputSingle(swap_params_struct).build_transaction,
                tx_params,
            )
            signed_tx = self._sign_transaction(swap_tx)
            tx_hash = await self._rpc(
                params.chain, w3.eth.send_raw_transaction, signed_tx.raw_transaction
            )
            logger.info(f"DEX swap transaction sent: {tx_hash.hex()}")

            receipt = await self._rpc_unpaced(
                w3.eth.wait_for_transaction_receipt, tx_hash,
                timeout=self.net_config.max_pending_seconds,
            )
            
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
