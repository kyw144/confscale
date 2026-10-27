"""Supported method capabilities and explicit comparison presets."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Capabilities:
    native_hpa: bool = False
    forecasts: bool = False
    interval_monitor: bool = False
    model_head: str | None = None
    target_rule: str | None = None


LEGACY_METHODS = ('hpa-reactive', 'static-2', 'static-4', 'static-8', 'confscale-scp',
                  'confscale-be', 'confscale-qr', 'confscale-aci', 'confscale-pid',
                  'confscale-rolling-origin', 'confscale-rolling-origin-warmstart',
                  'confscale-pid-laddered', 'confscale-aci-laddered', 'confscale-rolling-origin-laddered')
COMPARISON_METHODS = ('confscale-point-corrected', 'confscale-scp-frozen',
                      'hpa-cpu-tuned', 'hpa-cpu-stock')
METHODS = LEGACY_METHODS + COMPARISON_METHODS
_CAPABILITIES = {
    name: Capabilities(native_hpa=name == 'hpa-reactive',
                       forecasts=name.startswith('confscale-'),
                       interval_monitor=name.startswith('confscale-'),
                       model_head=(name.removeprefix('confscale-') if name in ('confscale-be', 'confscale-qr')
                                   else 'scp' if name.startswith('confscale-') else None))
    for name in LEGACY_METHODS
}
_CAPABILITIES.update({
    'confscale-point-corrected': Capabilities(forecasts=True, model_head='scp', target_rule='corrected-h0'),
    'confscale-scp-frozen': Capabilities(forecasts=True, model_head='scp', target_rule='frozen-tier'),
    'hpa-cpu-tuned': Capabilities(native_hpa=True),
    'hpa-cpu-stock': Capabilities(native_hpa=True),
})


def capabilities(name):
    try:
        return _CAPABILITIES[name]
    except KeyError as error:
        raise ValueError(f'Unsupported method: {name}') from error


def capability_record(name):
    return asdict(capabilities(name))


def preset_configuration(name):
    """Settings that define the named comparison, independent of runtime objects."""
    if name in ('confscale-point-corrected', 'confscale-scp-frozen'):
        return {'uq_method': 'scp', 'policy': 'tier', 'slo_capacity': 10.0,
                'target_util': 0.7, 'safety_factor': 2.0, 'min_replicas': 1, 'max_replicas': 20,
                'coverage_monitor': False, 'online_recal': False, 'recalibrator': 'none',
                'ladder': False, 'lambda_risk': None,
                'h0_only': name == 'confscale-point-corrected',
                'point_bias_window': 10 if name == 'confscale-point-corrected' else 0}
    if name in ('hpa-cpu-tuned', 'hpa-cpu-stock'):
        return {'cpu_target_pct': 70 if name == 'hpa-cpu-tuned' else 80,
                'min_replicas': 1, 'max_replicas': 20,
                'behavior': 'percent-100-per-15s-downscale-60s' if name == 'hpa-cpu-tuned' else 'omitted'}
    return {}


def configure_method(method):
    """Preserve legacy monitor enablement; enforce explicit new preset settings."""
    cap = capabilities(method.name)
    if hasattr(method, 'coverage_monitor'):
        method.coverage_monitor = cap.interval_monitor
    if cap.target_rule:
        for key, value in preset_configuration(method.name).items():
            setattr(method, key, value)
    return method
