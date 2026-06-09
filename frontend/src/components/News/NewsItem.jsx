import { ExternalLink } from "lucide-react";

const SENTIMENT_STYLE = {
  positive: { label: "Bullish", cls: "bg-emerald-400/10 text-emerald-400" },
  negative: { label: "Bearish", cls: "bg-red-400/10 text-red-400"         },
  neutral:  { label: "Neutral", cls: "bg-slate-700/50 text-slate-400"     },
};

export default function NewsItem({ item }) {
  const s   = SENTIMENT_STYLE[item.sentiment] || SENTIMENT_STYLE.neutral;
  const ts  = item.published_at
    ? new Date(item.published_at).toLocaleDateString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })
    : "";

  return (
    <div className="border-b border-surface-border py-4 hover:bg-surface-hover -mx-4 px-4 transition-colors">
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1.5 flex-wrap">
            {item.ticker && (
              <span className="text-xs font-mono font-semibold text-brand bg-brand-dim px-2 py-0.5 rounded">
                {item.ticker}
              </span>
            )}
            <span className={`text-xs px-2 py-0.5 rounded font-medium ${s.cls}`}>{s.label}</span>
            <span className="text-xs text-slate-600">{item.source}</span>
          </div>

          <p className="text-sm text-white font-medium leading-snug mb-1 line-clamp-2">
            {item.headline}
          </p>

          {item.summary && (
            <p className="text-xs text-slate-500 leading-relaxed line-clamp-2">{item.summary}</p>
          )}

          <p className="text-xs text-slate-600 mt-2">{ts}</p>
        </div>

        {item.url && (
          <a
            href={item.url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex-shrink-0 text-slate-600 hover:text-brand transition-colors mt-0.5"
          >
            <ExternalLink className="w-4 h-4" />
          </a>
        )}
      </div>
    </div>
  );
}
