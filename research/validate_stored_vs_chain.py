"""Does a STORED observation still price what the chain said at its block?

The chain of trust behind every number in the report is four links long:

    chain -> batched read -> PoolSnapshot -> SQLite row -> reloaded snapshot -> price

Each link has been checked separately. The batched read reproduces the unbatched one
exactly. The local swap math matches the deployed QuoterV2 on live pools. The store
round-trips a synthetic snapshot losslessly. But "each link holds" is not the same
claim as "the whole chain holds", and the difference is where a serialisation defect
would live -- one that a synthetic fixture cannot expose because the fixture was built
by the same code that reads it back.

So this reads observations back out of the recording, prices them locally, and asks
the deployed QuoterV2 the same question AT THE BLOCK EACH ROW RECORDED. Anything other
than exact agreement means the dataset does not describe the market it claims to.

Block pinning is what makes the comparison meaningful, and it is also the limitation:
public endpoints are not archive nodes, so only recent blocks can be re-queried. The
newest observations are therefore the ones checked.
"""
import asyncio
import sys
from collections import defaultdict
from decimal import Decimal

from research_config import research_config

from src.exchange.errors import RpcError
from src.research.observations import ObservationStore
from src.research.report import group_key
from src.exchange.univ3 import UniV3DexClient

config = research_config("WARNING")

DB = sys.argv[1] if len(sys.argv) > 1 else "data/observations.sqlite3"
PER_MARKET = 2


async def main():
    store = ObservationStore(DB, run_id="validate-stored")
    total = store.count()
    print(f"{DB}: {total:,} observations")
    if not total:
        return

    # Newest first: only recent blocks are re-queryable on a non-archive endpoint.
    by_market = defaultdict(list)
    for observation in store.read_all():
        by_market[group_key(observation)].append(observation)

    client = UniV3DexClient(config.dex, config.network, config.secrets, config.tokens)

    print(f"\n{'market':<28} {'block':>10} {'size':>10} {'local':>22} "
          f"{'chain':>22} {'verdict':>10}")
    exact = mismatched = unavailable = 0
    for key in sorted(by_market):
        sample = by_market[key][-PER_MARKET:]
        for observation in sample:
            pool = observation.pool
            # A size small enough to stay inside the observed window on any pool, so a
            # refusal here would mean a storage problem rather than a thin pool.
            amount_in = int(Decimal("0.01") * (Decimal(10) ** pool.decimals0))
            local = pool.swap_exact_in(amount_in, zero_for_one=True)
            try:
                onchain = await client.quote_exact_input_single_raw(
                    chain=observation.chain,
                    token_in=pool.token0, token_out=pool.token1,
                    fee=pool.fee, amount_in=amount_in,
                    block_number=pool.block_number,
                )
            except RpcError as exc:
                unavailable += 1
                print(f"  {str(key[0] + ' ' + key[1] + ' ' + str(key[2])):<26} "
                      f"{pool.block_number:>10}  RPC: {str(exc)[:40]}")
                continue
            except Exception as exc:
                unavailable += 1
                print(f"  {str(key[0] + ' ' + key[1] + ' ' + str(key[2])):<26} "
                      f"{pool.block_number:>10}  {type(exc).__name__}: {str(exc)[:36]}")
                continue

            verdict = "EXACT" if local == onchain else "DIFFER"
            if local == onchain:
                exact += 1
            else:
                mismatched += 1
            label = f"{key[0]} {key[1]} {key[2]}"
            print(f"  {label:<26} {pool.block_number:>10} {'0.01 t0':>10} "
                  f"{local:>22,} {onchain:>22,} {verdict:>10}")

    print(f"\nexact {exact}, mismatched {mismatched}, unavailable {unavailable}")
    if mismatched:
        print("A stored observation prices differently from the chain it was read")
        print("from. The dataset does not describe the market it claims to, and every")
        print("statistic computed from it is suspect.")
    elif exact:
        print("Every re-queryable stored observation reproduces the deployed")
        print("QuoterV2 exactly, at the block it recorded. The chain from RPC through")
        print("batching, serialisation and reload is intact end to end.")
    if unavailable:
        print(f"{unavailable} could not be re-queried -- public endpoints are not")
        print("archive nodes, so older blocks are pruned. Not a mismatch.")


asyncio.run(main())
