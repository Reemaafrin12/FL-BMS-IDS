import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const formatConfidence = (value) => `${(Number(value) * 100).toFixed(1)}%`;
const labelOf = (reading) => String(reading.prediction || "").toLowerCase();

export default function App() {
  const [readings, setReadings] = useState([]);
  const [error, setError] = useState("");
  const [connected, setConnected] = useState(false);
  const [clock, setClock] = useState(new Date());
  const [filter, setFilter] = useState("all");
  const [newIds, setNewIds] = useState([]);

  useEffect(() => {
    let previousIds = new Set();
    const load = async () => {
      try {
        const response = await fetch("/api/readings/recent");
        if (!response.ok) throw new Error("Could not load readings");
        const next = await response.json();
        const arrived = next.filter((reading) => !previousIds.has(reading._id)).map((reading) => reading._id);
        if (previousIds.size) setNewIds(arrived);
        previousIds = new Set(next.map((reading) => reading._id));
        setReadings(next);
        setError("");
        setConnected(true);
      } catch (err) {
        setError(err.message);
        setConnected(false);
      }
    };
    load();
    const pollId = setInterval(load, 3000);
    const clockId = setInterval(() => setClock(new Date()), 1000);
    return () => {
      clearInterval(pollId);
      clearInterval(clockId);
    };
  }, []);

  const benignCount = readings.filter((reading) => labelOf(reading) === "benign").length;
  const attackCount = readings.filter((reading) => labelOf(reading) === "attack").length;
  const attackRate = readings.length ? (attackCount / readings.length) * 100 : 0;
  const recentMix = useMemo(() => {
    const lastTwenty = readings.slice(0, 20);
    return [
      { type: "Benign", count: lastTwenty.filter((reading) => labelOf(reading) === "benign").length, color: "#16a37a" },
      { type: "Attack", count: lastTwenty.filter((reading) => labelOf(reading) === "attack").length, color: "#e45454" },
    ];
  }, [readings]);
  const visibleReadings = readings.filter((reading) => filter === "all" || labelOf(reading) === filter);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark">EV</div>
          <div><p className="eyebrow">FLEET SECURITY / MONITORING</p><h1>EVSE IDS</h1></div>
        </div>
        <div className="header-meta">
          <span className={`connection ${connected ? "is-connected" : "is-disconnected"}`}>
            <i />{connected ? "Connected" : "Disconnected"}
          </span>
          <span className="live-clock">{clock.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}</span>
        </div>
      </header>

      <section className="intro-row">
        <div><p className="eyebrow">LIVE OVERVIEW</p><h2>Charging network activity</h2><p className="muted">Intrusion predictions refresh every 3 seconds.</p></div>
        <div className="feed-indicator"><span className="feed-dot" /> REAL-TIME FEED</div>
      </section>

      {error && <div className="error-banner" role="alert">{error} · Retrying automatically</div>}

      <section className="stats-grid" aria-label="Reading summary">
        <article className="stat-card"><span className="stat-label">Total readings</span><strong>{readings.length}</strong><span className="stat-foot">Latest records in view</span></article>
        <article className="stat-card benign-stat"><span className="stat-label">Benign count</span><strong>{benignCount}</strong><span className="stat-foot"><i /> Normal activity</span></article>
        <article className="stat-card attack-stat"><span className="stat-label">Attack count</span><strong>{attackCount}</strong><span className="stat-foot"><i /> Threat signals</span></article>
        <article className="stat-card rate-stat"><span className="stat-label">Attack rate</span><strong>{attackRate.toFixed(1)}<small>%</small></strong><span className="stat-foot">Of fetched readings</span></article>
      </section>

      <section className="chart-card">
        <div className="section-heading"><div><p className="eyebrow">LAST 20 READINGS</p><h3>Prediction mix</h3></div><span className="chart-caption">Rolling class count</span></div>
        <div className="chart-wrap">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={recentMix} margin={{ top: 8, right: 20, left: -22, bottom: 0 }} barSize={42}>
              <CartesianGrid strokeDasharray="3 6" vertical={false} stroke="#e8edf2" />
              <XAxis dataKey="type" axisLine={false} tickLine={false} tick={{ fill: "#718096", fontSize: 12 }} />
              <YAxis allowDecimals={false} axisLine={false} tickLine={false} tick={{ fill: "#98a2b3", fontSize: 11 }} />
              <Tooltip cursor={{ fill: "#f5f7fa" }} contentStyle={{ border: "1px solid #e7ecf1", borderRadius: 10, boxShadow: "0 8px 24px #1d293910" }} />
              <Bar dataKey="count" radius={[7, 7, 2, 2]}>
                {recentMix.map((entry) => <Cell key={entry.type} fill={entry.color} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="table-card">
        <div className="table-toolbar">
          <div><p className="eyebrow">DETECTION LOG</p><h3>Recent predictions</h3></div>
          <div className="filter-group" role="group" aria-label="Filter readings">
            {[ ["all", "All"], ["attack", "Attacks only"], ["benign", "Benign only"] ].map(([value, text]) => (
              <button key={value} className={filter === value ? "active" : ""} onClick={() => setFilter(value)}>{text}</button>
            ))}
          </div>
        </div>
        <div className="table-scroll"><table>
          <thead><tr><th>Timestamp</th><th>Current</th><th>Power</th><th>State</th><th>Prediction</th><th>Confidence</th></tr></thead>
          <tbody>{visibleReadings.map((reading) => {
            const attack = labelOf(reading) === "attack";
            return <tr key={reading._id} className={`${attack ? "row-attack" : "row-benign"} ${newIds.includes(reading._id) ? "new-reading" : ""}`}>
              <td className="timestamp">{new Date(reading.timestamp).toLocaleString()}</td>
              <td>{reading.current_mA}<span className="unit"> mA</span></td><td>{reading.power_mW}<span className="unit"> mW</span></td><td>{reading.State}</td>
              <td><span className={`prediction-pill ${attack ? "pill-attack" : "pill-benign"}`}><span aria-hidden="true">{attack ? "⚠" : "✓"}</span>{reading.prediction}</span></td>
              <td className="confidence">{formatConfidence(reading.confidence)}</td>
            </tr>;
          })}</tbody>
        </table>{visibleReadings.length === 0 && <div className="empty-state">No readings match this filter yet.</div>}</div>
        <footer className="table-footer">Showing {visibleReadings.length} of {readings.length} readings <span>Updates automatically</span></footer>
      </section>
      <footer className="page-footer">EVSE intrusion detection <span>·</span> Secure charging operations</footer>
    </main>
  );
}
