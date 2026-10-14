import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { App } from "@/App";
import { env } from "@/config/env";
import "@/styles/globals.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter basename={env.basePath === "/" ? undefined : env.basePath}>
      <App />
    </BrowserRouter>
  </StrictMode>,
);
