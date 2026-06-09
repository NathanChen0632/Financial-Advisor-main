import { StrictMode, Component } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.jsx";
import { AuthProvider } from "./contexts/AuthContext.jsx";

// Catches any render-time crash and shows the error on screen
// instead of a blank white page — helps diagnose setup issues.
class ErrorBoundary extends Component {
  constructor(props) { super(props); this.state = { error: null }; }
  static getDerivedStateFromError(e) { return { error: e }; }
  render() {
    if (this.state.error) {
      return (
        <div style={{
          padding: "2rem", fontFamily: "monospace", background: "#0f1117",
          color: "#f87171", minHeight: "100vh"
        }}>
          <h2 style={{ color: "#fff", marginBottom: "1rem" }}>App crashed — check the error below:</h2>
          <pre style={{ whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
            {this.state.error.toString()}
            {"\n\n"}
            {this.state.error.stack}
          </pre>
          <p style={{ color: "#94a3b8", marginTop: "1.5rem" }}>
            Common fixes: run <code style={{color:"#00d4aa"}}>npm install</code> in the frontend folder,
            or fill in your Supabase keys in <code style={{color:"#00d4aa"}}>frontend/.env.local</code>
          </p>
        </div>
      );
    }
    return this.props.children;
  }
}

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <ErrorBoundary>
      <AuthProvider>
        <App />
      </AuthProvider>
    </ErrorBoundary>
  </StrictMode>
);
