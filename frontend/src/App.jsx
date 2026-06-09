import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { useAuth }    from "./contexts/AuthContext";
import Login          from "./pages/Login";
import Sidebar        from "./components/Layout/Sidebar";
import Dashboard      from "./pages/Dashboard";
import Portfolio      from "./pages/Portfolio";
import Recommendations from "./pages/Recommendations";
import Performance    from "./pages/Performance";
import News           from "./pages/News";
import Holdings       from "./pages/Holdings";
import Watchlist      from "./pages/Watchlist";
import Trade          from "./pages/Trade";
import Approvals      from "./pages/Approvals";

export default function App() {
  const { session, loading, isMaster } = useAuth();

  // While the initial session check is in flight, avoid flashing the login page.
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-surface text-slate-500 text-sm">
        Loading…
      </div>
    );
  }

  // No session → everything routes to the login screen.
  if (!session) return <Login />;

  return (
    <BrowserRouter>
      <div className="flex min-h-screen bg-surface text-slate-300">
        <Sidebar />

        {/* Main content — offset by sidebar width */}
        <main className="flex-1 ml-56 p-7 max-w-6xl">
          <Routes>
            <Route index                  element={<Dashboard />}       />
            <Route path="/recommendations" element={<Recommendations />} />
            <Route path="/news"            element={<News />}            />

            {/* Master-only: live Alpaca account + trading */}
            {isMaster && <Route path="/portfolio"   element={<Portfolio />}   />}
            {isMaster && <Route path="/trade"       element={<Trade />}       />}
            {isMaster && <Route path="/approvals"   element={<Approvals />}   />}
            {isMaster && <Route path="/performance" element={<Performance />} />}

            {/* Suggestion-only users: paper portfolio + watchlist */}
            {!isMaster && <Route path="/holdings"  element={<Holdings />}  />}
            {!isMaster && <Route path="/watchlist" element={<Watchlist />} />}

            {/* Unknown routes fall back to the dashboard */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
