import ReactMarkdown from "react-markdown";

type Props = {
  report?: string;
  sample?: Array<{ id: string; number: string; title: string }>;
  debug?: any;
  loading?: boolean;
  error?: string | null;
};

export default function ReportView({ report, sample, debug, loading, error }: Props) {
  return (
    <div className="card" style={{ marginTop: 16 }}>
      <h3>דו"ח</h3>
      {loading && <p>מייצר דו"ח...</p>}
      {error && <p style={{ color: "crimson" }}>{String(error)}</p>}

      {report ? (
        <div className="markdown">
          <ReactMarkdown>{report}</ReactMarkdown>
        </div>
      ) : (
        !loading && <p className="muted">טרם הופק דו"ח.</p>
      )}

      {sample && sample.length > 0 && (
        <>
          <div className="hr" />
          <h4>דוגמת כללים שנשלחו ל-LLM</h4>
          <ul className="list">
            {sample.map((r) => (
              <li key={r.id}>
                <strong>{r.number}</strong> — {r.title}
              </li>
            ))}
          </ul>
        </>
      )}

      {debug && (
        <>
          <div className="hr" />
          <h4>Debug</h4>
          <pre>{JSON.stringify(debug, null, 2)}</pre>
        </>
      )}
    </div>
  );
}
