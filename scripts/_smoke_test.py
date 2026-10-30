"""Quick smoke test of replay forecast + observed lookup."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.forecast import replay_forecast, observed_at

r = replay_forecast('2023-09-29 18:00')
print('Forecast time:', r.forecast_time)
print('Last input:', r.last_input_time)
print('Forecast (m):', r.forecast_value_m)
print('80% PI:', r.lo80, r.hi80)
print('95% PI:', r.lo95, r.hi95)
print('Status:', r.status)
print('Observed at forecast time:', observed_at(r.forecast_time))
