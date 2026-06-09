import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  Briefcase,
  Zap,
  TrendingUp,
  Newspaper,
  Activity,
  Wallet,
  Eye,
  ArrowLeftRight,
  CheckSquare,
  LogOut,
} from "lucide-react";
import { useAuth } from "../../contexts/AuthContext";

// Routes shown to everyone.
const COMMON = [
  { to: "/",                icon: LayoutDashboard, label: "Dashboard"       },
  { to: "/recommendations", icon: Zap,             label: "Recommendations" },
  { to: "/news",            icon: Newspaper,       label: "News"            },
];

// Master (real Alpaca trading) sees live account + trade controls.
const MASTER = [
  { to: "/portfolio",   icon: Briefcase,      label: "Portfolio"   },
  { to: "/trade",       icon: ArrowLeftRight, label: "Trade"       },
  { to: "/approvals",   icon: CheckSquare,    label: "Approvals"   },
  { to: "/performance", icon: TrendingUp,     label: "Performance" },
];

// Suggestion-only users manage a paper portfolio + watchlist.
const USER = [
  { to: "/holdings",  icon: Wallet, label: "My Holdings" },
  { to: "/watchlist", icon: Eye,    label: "Watchlist"   },
];

export default function Sidebar() {
  const { isMaster, user, signOut } = useAuth();
  const nav = [...COMMON, ...(isMaster ? MASTER : USER)];

  return (
    <aside className="fixed left-0 top-0 h-screen w-56 bg-surface-card border-r border-surface-border flex flex-col z-10">
      {/* Logo */}
      <div className="px-5 py-6 border-b border-surface-border">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-brand" />
          <span className="font-semibold text-white text-sm tracking-wide">DQN Trader</span>
        </div>
        <p className="text-xs text-slate-500 mt-1">
          {isMaster ? "Live trading account" : "AI-powered suggestions"}
        </p>
      </div>

      {/* Nav links */}
      <nav className="flex-1 px-3 py-4 space-y-1">
        {nav.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all ${
                isActive
                  ? "bg-brand-dim text-brand font-medium"
                  : "text-slate-400 hover:text-white hover:bg-surface-hover"
              }`
            }
          >
            <Icon className="w-4 h-4 flex-shrink-0" />
            {label}
          </NavLink>
        ))}
      </nav>

      {/* Account / sign out */}
      <div className="px-3 py-4 border-t border-surface-border space-y-2">
        <div className="px-2">
          <p className="text-xs text-slate-300 truncate">{user?.email}</p>
          <p className="text-[11px] text-slate-500">{isMaster ? "Master · trades live" : "Suggestions only"}</p>
        </div>
        <button
          onClick={signOut}
          className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-slate-400 hover:text-white hover:bg-surface-hover transition-all"
        >
          <LogOut className="w-4 h-4" />
          Sign out
        </button>
      </div>
    </aside>
  );
}
