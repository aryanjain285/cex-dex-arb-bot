"""Does the screen's derived token order match what each pool actually reports?

The screen derives `base_is_token0` from address ordering, because Uniswap v3 sorts a
pool's tokens by address and the addresses are already known -- so reading token0() per
pool per cycle would spend a call to learn a constant.

Deriving it is correct in principle. The consequence of being wrong is a factor of
price squared, which on a $2 token is a dislocation of tens of thousands of basis
points and on a token near $1 is almost invisible. So it is checked against the chain
once, per pool, rather than trusted.

Also flagged: any recorded dislocation beyond a plausibility bound. A real CEX-DEX gap
on a liquid asset is single-digit to low-double-digit basis points. Hundreds of bps is
a thin pool; thousands is an inverted price or a different asset entirely. The screen's
identity guards should have prevented the latter, and this is where that claim is
tested against data rather than asserted.
"""
import asyncio
import json
import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from research_config import research_config
from web3 import Web3

from src.exchange.multicall import Multicall
from src.exchange.pool_state import POOL_ABI
from src.exchange.univ3 import UniV3DexClient
from src.research.observations import ObservationStore, mid_dislocation_bps
from src.research.report import group_key

config = research_config("WARNING")

DB = sys.argv[1] if len(sys.argv) > 1 else "data/screen.sqlite3"
TARGETS = Path("research/targets_wide.json")

# Above this, a "dislocation" is almost certainly not a price difference between two
# venues quoting the same asset. Deliberately generous: thin pools genuinely produce
# hundreds of bps of impact, and the point is to catch inversions, not thinness.
IMPLAUSIBLE_BPS = Decimal("2000")


def encode(contract, name):
    encoder = getattr(contract, "encode_abi", None)
    if encoder is not None:
        try:
            return encoder(abi_element_identifier=name, args=[])
        except TypeError:
            return encoder(name, [])
    return contract.encodeABI(fn_name=name, args=[])


async def main():
    store = ObservationStore(DB, run_id="validate-orientation")
    total = store.count()
    print(f"{DB}: {total:,} observations")
    if not total:
        return

    targets = {
        (t["cex_symbol"], t["chain"], t["fee"]): t
        for t in json.loads(TARGETS.read_text(encoding="utf-8"))
    }

    # One observation per market is enough: token order is a property of the pool.
    latest = {}
    dislocations = defaultdict(list)
    for observation in store.read_all():
        key = group_key(observation)
        latest[key] = observation
        target = targets.get(key)
        if target is None:
            continue
        base_is_token0 = (
            str(target["base_address"]).lower() < str(target["quote_address"]).lower()
        )
        value = mid_dislocation_bps(observation, base_is_token0)
        if value is not None:
            dislocations[key].append(value)

    client = UniV3DexClient(config.dex, config.network, config.secrets, config.tokens)
    multicall = Multicall(client)

    by_chain = defaultdict(list)
    for key, observation in latest.items():
        by_chain[observation.chain].append((key, observation))

    wrong = 0
    checked = 0
    for chain, entries in by_chain.items():
        w3 = client._get_w3(chain)
        template = w3.eth.contract(
            address=Web3.to_checksum_address(entries[0][1].pool_address), abi=POOL_ABI
        )
        token0_data = encode(template, "token0")
        calls = [
            (Web3.to_checksum_address(o.pool_address), token0_data)
            for _, o in entries
        ]
        if await multicall.available(chain):
            raw = await multicall.aggregate(chain, calls)
        else:
            raw = []
            for _, o in entries:
                contract = w3.eth.contract(
                    address=Web3.to_checksum_address(o.pool_address), abi=POOL_ABI
                )
                try:
                    raw.append(await client._rpc(chain, contract.functions.token0().call))
                except Exception:
                    raw.append(None)

        for (key, observation), data in zip(entries, raw):
            if data is None:
                continue
            try:
                actual = (
                    w3.codec.decode(["address"], data)[0]
                    if isinstance(data, (bytes, bytearray)) else str(data)
                )
            except Exception:
                continue
            checked += 1
            target = targets.get(key)
            if target is None:
                continue
            derived_token0 = (
                target["base_address"]
                if str(target["base_address"]).lower() < str(target["quote_address"]).lower()
                else target["quote_address"]
            )
            if str(actual).lower() != str(derived_token0).lower():
                wrong += 1
                print(f"  ORDER WRONG {key}: chain token0 {actual}, "
                      f"derived {derived_token0}")

    print(f"\ntoken order: {checked} pools checked, {wrong} wrong")
    if wrong == 0 and checked:
        print("  Address ordering matches every pool the chain reports. Deriving it")
        print("  rather than reading it per cycle is safe.")

    print("\nimplausible dislocations (a price inversion or a different asset):")
    flagged = 0
    for key in sorted(dislocations):
        values = dislocations[key]
        worst = max(values, key=abs)
        if abs(worst) > IMPLAUSIBLE_BPS:
            flagged += 1
            print(f"  {key[0]:<14} {key[1]:<9} {key[2]:>5}  "
                  f"worst {float(worst):>14,.0f} bps  n={len(values)}")
    if not flagged:
        print(f"  none beyond {IMPLAUSIBLE_BPS} bps across {len(dislocations)} markets")

    print("\nlargest |dislocation| per market, top 25:")
    ranked = sorted(
        ((max((abs(v) for v in vs), default=Decimal(0)), key) for key, vs in dislocations.items()),
        reverse=True,
    )
    for worst, key in ranked[:25]:
        values = dislocations[key]
        signs = {v > 0 for v in values}
        kind = "one-sided" if len(signs) == 1 else "BOTH SIGNS"
        print(f"  {key[0]:<14} {key[1]:<9} {key[2]:>5}  max |d| "
              f"{float(worst):>10,.1f} bps  n={len(values):>4}  {kind}")


asyncio.run(main())
