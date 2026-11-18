/** Variable selection presets for ablation datasets (which Table 3 columns a dataset drops). */

/** Bureau scores: FICO at application and its interim refresh. */
export const FICO_VARIABLES = ['fico_range_low', 'fico_range_high', 'last_fico_range_low', 'last_fico_range_high'];

/** Lending Club's own risk grade, priced largely off FICO: a second route for the score to leak in. */
export const LC_GRADE_VARIABLES = ['grade', 'sub_grade'];

/** Credit-report tradeline attributes (what a thin-file applicant does not have). */
export const BUREAU_VARIABLES = [
  'delinq_2yrs',
  'delinq_amnt',
  'inq_last_6mths',
  'inq_last_12m',
  'open_acc',
  'total_acc',
  'revol_bal',
  'revol_util',
  'mort_acc',
  'pub_rec',
  'pub_rec_bankruptcies',
  'collections_12_mths_ex_med',
  'num_accts_ever_120_pd',
  'pct_tl_nvr_dlq',
  'num_sats',
  'tot_coll_amt',
  'tot_cur_bal',
  'tot_hi_cred_lim',
  'total_bal_ex_mort',
  'total_bc_limit',
  'bc_util',
  'all_util',
  'num_tl_30dpd',
  'num_tl_90g_dpd_24m',
  'mths_since_last_delinq',
];

export interface VariablePreset {
  key: string;
  label: string;
  blurb: string;
  exclude: string[];
}

export const VARIABLE_PRESETS: VariablePreset[] = [
  { key: 'all', label: 'All variables', blurb: 'The full Table 3 set (baseline).', exclude: [] },
  { key: 'no_fico', label: 'No FICO', blurb: 'Drop the FICO range at application and its interim refresh.', exclude: FICO_VARIABLES },
  {
    key: 'no_fico_grade',
    label: 'No FICO, no LC grade',
    blurb: 'Also drop grade / sub-grade, which Lending Club priced largely off FICO.',
    exclude: [...FICO_VARIABLES, ...LC_GRADE_VARIABLES],
  },
  {
    key: 'thin_file',
    label: 'Thin file',
    blurb: 'Application data only: no FICO, no LC grade, no credit-report tradelines.',
    exclude: [...FICO_VARIABLES, ...LC_GRADE_VARIABLES, ...BUREAU_VARIABLES],
  },
];

const sameSet = (a: string[], b: string[]) => a.length === b.length && a.every((x) => b.includes(x));

/** The preset an exclusion list matches exactly (restricted to the selectable names), if any. */
export function matchingPreset(excluded: string[], selectable: string[]): VariablePreset | undefined {
  return VARIABLE_PRESETS.find((p) => sameSet(excluded, p.exclude.filter((x) => selectable.includes(x))));
}

/** Short tags for a dataset card: "no FICO", "no LC grade", "−n vars". */
export function ablationTags(excluded: string[] | null | undefined): string[] {
  const ex = excluded ?? [];
  if (!ex.length) return [];
  const tags: string[] = [];
  if (FICO_VARIABLES.slice(0, 2).every((x) => ex.includes(x))) tags.push('no FICO');
  if (LC_GRADE_VARIABLES.every((x) => ex.includes(x))) tags.push('no LC grade');
  tags.push(`−${ex.length} var${ex.length === 1 ? '' : 's'}`);
  return tags;
}
