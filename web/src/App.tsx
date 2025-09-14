import { useEffect, useState } from "react";
import ReportForm from "./components/ReportForm";
import ReportView from "./components/ReportView";
import { health, postReport, type ReportReq, type ReportResp } from "./lib/api";

export default function App() {
  const [alive, setAlive] = useState<null | { ok: boolean; rules_loaded: number; feature_index: boolean }>(null);
  const [loading, setLoading] = useState(false);
  const [resp, setResp] = useState<ReportResp | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    health()
      .then(setAlive)
      .catch((e) => setErr(String(e)));
  }, []);

  async function handleSubmit(p: ReportReq) {
    setLoading(true);
    setErr(null);
    setResp(null);
    try {
      const r = await postReport(p);
      setResp(r);
    } catch (e: any) {
      setErr(e?.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="container">
      <h2>BLA — Business License Advisor</h2>
      <p className="muted">
        מצב שרת:{" "}
        {alive ? (
          alive.ok ? (
            <>
              ✅ rules:{alive.rules_loaded} · feature index:{" "}
              {alive.feature_index ? "✅" : "—"}
            </>
          ) : (
            "server responded but ok=false"
          )
        ) : (
          "בודק חיבור..."
        )}
      </p>

      <ReportForm onSubmit={handleSubmit} loading={loading} />

      <ReportView
        loading={loading}
        error={err}
        report={resp?.report}
        sample={resp?.sample_rules}
        debug={resp?.debug_info}
      />
    </div>
  );
}
