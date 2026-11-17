import { createContext, useContext, useState, type ReactNode } from "react";

export type Mode = "analyst" | "scientist";
const Ctx = createContext<{ mode: Mode; setMode: (m: Mode) => void }>({ mode: "analyst", setMode: () => {} });

function initial(): Mode {
  try { return (localStorage.getItem("aaml-mode") as Mode) || "analyst"; } catch { return "analyst"; }
}

export function ModeProvider({ children }: { children: ReactNode }) {
  const [mode, set] = useState<Mode>(initial);
  const setMode = (m: Mode) => { set(m); try { localStorage.setItem("aaml-mode", m); } catch { /* storage unavailable */ } };
  return <Ctx.Provider value={{ mode, setMode }}>{children}</Ctx.Provider>;
}

export const useMode = () => useContext(Ctx);
/** Content shown only in Data Scientist mode. */
export const DS = ({ children }: { children: ReactNode }) => (useMode().mode === "scientist" ? <>{children}</> : null);
/** Content shown only in Analyst mode. */
export const Analyst = ({ children }: { children: ReactNode }) => (useMode().mode === "analyst" ? <>{children}</> : null);
