import * as Tooltip from "@radix-ui/react-tooltip";
import { lazy, Suspense } from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { PageLoading } from "./components/ui";
import { ModeProvider } from "./lib/mode";

const Overview = lazy(() => import("./pages/Overview"));
const Queue = lazy(() => import("./pages/Queue"));
const AlertDetail = lazy(() => import("./pages/AlertDetail"));
const Clusters = lazy(() => import("./pages/Clusters"));
const Sequences = lazy(() => import("./pages/Sequences"));
const Arena = lazy(() => import("./pages/Arena"));
const Ensembles = lazy(() => import("./pages/Ensembles"));
const DataInsights = lazy(() => import("./pages/DataInsights"));
const About = lazy(() => import("./pages/About"));

export default function App() {
  return (
    <ModeProvider>
      <Tooltip.Provider delayDuration={150}>
        <BrowserRouter>
          <Layout>
            <Suspense fallback={<PageLoading />}>
              <Routes>
                <Route path="/" element={<Overview />} />
                <Route path="/queue" element={<Queue />} />
                <Route path="/alert" element={<AlertDetail />} />
                <Route path="/alert/:id" element={<AlertDetail />} />
                <Route path="/clusters" element={<Clusters />} />
                <Route path="/sequences" element={<Sequences />} />
                <Route path="/arena" element={<Arena />} />
                <Route path="/ensembles" element={<Ensembles />} />
                <Route path="/data" element={<DataInsights />} />
                <Route path="/about" element={<About />} />
                <Route path="*" element={<Overview />} />
              </Routes>
            </Suspense>
          </Layout>
        </BrowserRouter>
      </Tooltip.Provider>
    </ModeProvider>
  );
}
