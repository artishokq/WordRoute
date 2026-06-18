export interface DonorCandidate {
  language: string;
  probability: number;
}

export interface GlottologInfo {
  glottocode?: string;
  family?: string;
  subfamily?: string;
  macroarea?: string;
  latitude?: number;
  longitude?: number;
  country?: string;
  iso639?: string;
  description?: string;
}

export interface WordResult {
  word: string;
  lemma: string;
  pos?: string;
  gender?: string;
  is_declinable: boolean;
  loanword_probability: number;
  is_loanword: boolean;
  donor_language: string;
  donor_language_ru?: string;
  donor_family?: string;
  donor_subfamily?: string;
  source_word?: string;
  semantic_domain?: string;
  semantic_domain_ru?: string;
  explanation: string[];
  morphological_derivatives: string[];
  top_donors: DonorCandidate[];
  glottolog?: GlottologInfo;
  in_seed: boolean;
}

export interface AnalysisStats {
  total_words: number;
  borrowings_found: number;
  native_words: number;
  borrowing_rate: number;
  top_donor?: string;
  donor_distribution: Record<string, number>;
  semantic_distribution: Record<string, number>;
}

export interface AnalyzeResponse {
  input_text: string;
  mode: string;
  words: WordResult[];
  stats: AnalysisStats;
}

export type AnalysisMode = "word" | "text";
