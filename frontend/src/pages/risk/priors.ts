/** Form state of the fit form's "Priors & guardrails" section and its FitCreate payload. */
import type { ArmSpaceConfig, FitCreate, Guardrails, PolicyEffectFeature } from '@/api/types';
import { DEFAULT_FIT_CONFIG, GUARDRAIL_INFO } from '@/lib/domain';

export type GuardrailKey = keyof Guardrails;

export interface PriorsState {
  policyEffect: boolean;
  features: PolicyEffectFeature[];
  ridge: number | null;
  maxShift: number | null;
  minRows: number | null;
  minElapsedRows: number | null;
  maxIter: number | null;
  ptiCap: number | null;
  termLevel: boolean;
  prepayment: boolean;
  /** Guardrail values in display units (percent, or percentage points for the corridor). */
  guard: Record<GuardrailKey, number | null>;
  guardOn: Record<GuardrailKey, boolean>;
}

const D = DEFAULT_FIT_CONFIG;
const toDisplay = (v: number | null) => (v === null ? null : Math.round(v * 1000) / 10);

export function defaultPriors(): PriorsState {
  const pe = D.hierarchy.policy_effect_cfg;
  const arm = D.arm_space;
  return {
    policyEffect: D.hierarchy.policy_effect,
    features: [...pe.features],
    ridge: pe.ridge,
    maxShift: pe.max_abs_logit_shift,
    minRows: pe.min_rows,
    minElapsedRows: pe.min_elapsed_rows,
    maxIter: pe.max_iter,
    ptiCap: pe.pti_cap,
    termLevel: D.recovery.term_level,
    prepayment: D.reward.prepayment_adjustment === 'empirical',
    guard: Object.fromEntries(GUARDRAIL_INFO.map((g) => [g.key, toDisplay(arm[g.key])])) as Record<GuardrailKey, number | null>,
    guardOn: Object.fromEntries(GUARDRAIL_INFO.map((g) => [g.key, arm[g.key] !== null])) as Record<GuardrailKey, boolean>,
  };
}

/** Section payload, merged by FitForm into the FitCreate body (a switched-off guardrail is sent as null). */
export function priorsPayload(s: PriorsState): {
  guardrails: Pick<ArmSpaceConfig, GuardrailKey>;
  hierarchy: Pick<NonNullable<FitCreate['hierarchy']>, 'policy_effect' | 'policy_effect_cfg'>;
  term_level: boolean;
  prepayment_adjustment: 'empirical' | null;
} {
  const pe = D.hierarchy.policy_effect_cfg;
  const guardrails = Object.fromEntries(
    GUARDRAIL_INFO.map((g) => {
      const v = s.guard[g.key];
      return [g.key, s.guardOn[g.key] && v !== null ? v / 100 : null];
    }),
  ) as Pick<ArmSpaceConfig, GuardrailKey>;
  return {
    guardrails,
    hierarchy: {
      policy_effect: s.policyEffect,
      policy_effect_cfg: {
        features: s.features,
        ridge: s.ridge ?? pe.ridge,
        max_abs_logit_shift: s.maxShift ?? pe.max_abs_logit_shift,
        min_rows: s.minRows ?? pe.min_rows,
        min_elapsed_rows: s.minElapsedRows ?? pe.min_elapsed_rows,
        max_iter: s.maxIter ?? pe.max_iter,
        pti_cap: s.ptiCap ?? pe.pti_cap,
      },
    },
    term_level: s.termLevel,
    prepayment_adjustment: s.prepayment ? 'empirical' : null,
  };
}
