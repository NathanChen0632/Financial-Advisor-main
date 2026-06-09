import { useState }  from "react";
import { useNews }   from "../hooks/useNews";
import Header        from "../components/Layout/Header";
import NewsItem      from "../components/News/NewsItem";
import { Search }    from "lucide-react";

const TICKERS = ["ALL", "AAPL", "MSFT", "TSLA", "NVDA", "GOOGL", "META", "AMZN"];

export default function News() {
  const [ticker, setTicker] = useState(null);
  const { news, loading, error, refresh } = useNews(ticker, 40);

  return (
    <div>
      <Header
        title="Market News"
        subtitle="Latest headlines from Yahoo Finance"
        onRefresh={refresh}
        loading={loading}
      />

      {error && (
        <div className="mb-4 bg-red-400/10 border border-red-400/30 text-red-400 text-sm rounded-lg px-4 py-3">
          {error}
        </div>
      )}

      {/* Ticker filter pills */}
      <div className="flex flex-wrap gap-2 mb-5">
        {TICKERS.map(t => (
          <button
            key={t}
            onClick={() => setTicker(t === "ALL" ? null : t)}
            className={`text-xs px-3 py-1.5 rounded-lg transition-all font-mono ${
              (t === "ALL" && !ticker) || t === ticker
                ? "bg-brand text-black font-semibold"
                : "bg-surface-card border border-surface-border text-slate-400 hover:text-white"
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {/* News feed */}
      <div className="bg-surface-card border border-surface-border rounded-xl px-5 py-1">
        {loading ? (
          <div className="space-y-4 py-4">
            {[1, 2, 3, 4, 5].map(i => (
              <div key={i} className="h-16 bg-surface-hover rounded-lg animate-pulse" />
            ))}
          </div>
        ) : news.length ? (
          news.map((item, i) => <NewsItem key={i} item={item} />)
        ) : (
          <div className="text-center py-12 text-slate-500 text-sm">
            <Search className="w-8 h-8 mx-auto mb-3 opacity-30" />
            <p>No news items yet.</p>
            <p className="text-xs mt-1">Headlines load automatically when the daily job runs. Hit Refresh in a moment.</p>
          </div>
        )}
      </div>
    </div>
  );
}
