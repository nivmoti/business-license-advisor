export const API_BASE =
  import.meta.env.VITE_API_BASE || "http://localhost:8000";

export type ReportReq = {
  size_m2: number;
  seats: number;
  features_text: string;
  top_n_rules?: number;
  debug?: boolean;
};

export type ReportResp = {
  selected_count: number;
  matches: Array<any>;
  sample_rules: Array<{ id: string; number: string; title: string }>;
  report: string;
  debug_info?: any;
};

export async function health() {
  const r = await fetch(`${API_BASE}/health`);
  if (!r.ok) throw new Error(`health failed: ${r.status}`);
  return r.json();
}

export async function postReport(payload: ReportReq): Promise<ReportResp> {
  const r = await fetch(`${API_BASE}/report`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) {
    const t = await r.text().catch(() => "");
    throw new Error(`report failed: ${r.status} ${t}`);
  }
  return r.json();
}
