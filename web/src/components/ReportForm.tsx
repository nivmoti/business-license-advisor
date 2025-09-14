import { useState } from "react";
import type { ReportReq } from "../lib/api";

type Props = {
  initial?: Partial<ReportReq>;
  onSubmit: (p: ReportReq) => void;
  loading?: boolean;
};

export default function ReportForm({ initial, onSubmit, loading }: Props) {
  const [size, setSize] = useState<number>(initial?.size_m2 ?? 120);
  const [seats, setSeats] = useState<number>(initial?.seats ?? 80);
  const [features, setFeatures] = useState<string>(
    initial?.features_text ?? "בלוני גז, טיגון צ'יפס, אין אלכוהול, מצלמות אבטחה"
  );
  const [topN, setTopN] = useState<number>(initial?.top_n_rules ?? 60);
  const [debug, setDebug] = useState<boolean>(initial?.debug ?? true);

  function submit(e: React.FormEvent) {
    e.preventDefault();
    onSubmit({
      size_m2: Number(size) || 0,
      seats: Number(seats) || 0,
      features_text: features || "",
      top_n_rules: Number(topN) || 60,
      debug,
    });
  }

  return (
    <form className="card" onSubmit={submit}>
      <div className="row">
        <div className="col">
          <label>גודל העסק (מ"ר)</label>
          <input
            type="number"
            min={0}
            value={size}
            onChange={(e) => setSize(Number(e.target.value))}
          />
        </div>
        <div className="col">
          <label>מספר מקומות ישיבה/תפוסה</label>
          <input
            type="number"
            min={0}
            value={seats}
            onChange={(e) => setSeats(Number(e.target.value))}
          />
        </div>
      </div>

      <div className="row" style={{ marginTop: 12 }}>
        <div className="col">
          <label>מאפיינים (טקסט חופשי)</label>
          <textarea
            placeholder="לדוגמה: בלוני גז, משלוחים, אין אלכוהול, מצלמות אבטחה"
            value={features}
            onChange={(e) => setFeatures(e.target.value)}
          />
          <div className="muted">
            אפשר בעברית/אנגלית; שלילות מזוהות אוטומטית (ללא/אין/בלי/אסור).
          </div>
        </div>
      </div>

      <div className="row" style={{ marginTop: 12 }}>
        <div className="col">
          <label>מס׳ כללים מקסימלי לדו"ח</label>
          <input
            type="number"
            min={10}
            max={200}
            value={topN}
            onChange={(e) => setTopN(Number(e.target.value))}
          />
        </div>
        <div className="col" style={{ display: "flex", alignItems: "flex-end" }}>
          <label>
            <input
              type="checkbox"
              checked={debug}
              onChange={(e) => setDebug(e.target.checked)}
            />{" "}
            Debug
          </label>
        </div>
      </div>

      <div style={{ marginTop: 16, display: "flex", gap: 8 }}>
        <button disabled={loading} type="submit">
          הפקת דו"ח
        </button>
        <span className="muted">
          ה-API: <span className="badge">{import.meta.env.VITE_API_BASE || "http://localhost:8000"}</span>
        </span>
      </div>
    </form>
  );
}
