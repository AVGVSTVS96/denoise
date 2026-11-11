import "@fontsource-variable/geist";
import { initBotId } from "botid/client/core";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./index.css";

initBotId({ protect: [{ path: "/api/training", method: "POST" }] });

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
