import { createMorph } from "../vendor/morphicons/dom.js";
import { PAGE_ICONS } from "./page-morphicon-icons.mjs";

const SVG_NAMESPACE = "http://www.w3.org/2000/svg";

// Lucide icon data stays as data, which is the input format Morphicons expects.
const COMPASS = [
  ["circle", { cx: "12", cy: "12", r: "10" }],
  ["path", { d: "m16.24 7.76-1.804 5.411a2 2 0 0 1-1.265 1.265L7.76 16.24l1.804-5.411a2 2 0 0 1 1.265-1.265z" }],
];
const ROCKET = [
  ["path", { d: "M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5" }],
  ["path", { d: "M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09" }],
  ["path", { d: "M9 12a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.4 22.4 0 0 1-4 2z" }],
  ["path", { d: "M9 12H4s.55-3.03 2-4c1.62-1.08 5 .05 5 .05" }],
];
const TERMINAL = [
  ["path", { d: "M12 19h8" }],
  ["path", { d: "m4 17 6-6-6-6" }],
];
const FLASK = [
  ["path", { d: "M14 2v6a2 2 0 0 0 .245.96l5.51 10.08A2 2 0 0 1 18 22H6a2 2 0 0 1-1.755-2.96l5.51-10.08A2 2 0 0 0 10 8V2" }],
  ["path", { d: "M6.453 15h11.094" }],
  ["path", { d: "M8.5 2h7" }],
];
const GAUGE = [
  ["path", { d: "m12 14 4-4" }],
  ["path", { d: "M3.34 19a10 10 0 1 1 17.32 0" }],
];
const CODE = [
  ["path", { d: "m18 16 4-4-4-4" }],
  ["path", { d: "m6 8-4 4 4 4" }],
  ["path", { d: "m14.5 4-5 16" }],
];
const BOOK_OPEN = [
  ["path", { d: "M12 5v16" }],
  ["path", { d: "M20.001 19A2 2 0 0 0 22 17V5a2 2 0 0 0-1.999-2L16 3.002A5 5 0 0 0 12 5a5 5 0 0 0-4-2H4a2 2 0 0 0-2 2v12a2 2 0 0 0 1.999 2H8a5 5 0 0 1 4 2 5 5 0 0 1 4-2z" }],
];
const INFO = [
  ["circle", { cx: "12", cy: "12", r: "10" }],
  ["path", { d: "M12 16v-4" }],
  ["path", { d: "M12 8h.01" }],
];
const MENU = [
  ["path", { d: "M4 5h16" }],
  ["path", { d: "M4 12h16" }],
  ["path", { d: "M4 19h16" }],
];
const X = [
  ["path", { d: "M18 6 6 18" }],
  ["path", { d: "m6 6 12 12" }],
];
const SEARCH = [
  ["path", { d: "m21 21-4.34-4.34" }],
  ["circle", { cx: "11", cy: "11", r: "8" }],
];
const MOON = [
  ["path", { d: "M20.985 12.486a9 9 0 1 1-9.473-9.472c.405-.022.617.46.402.803a6 6 0 0 0 8.268 8.268c.344-.215.825-.004.803.401" }],
];
const SUN = [
  ["circle", { cx: "12", cy: "12", r: "4" }],
  ["path", { d: "M12 2v2" }],
  ["path", { d: "M12 20v2" }],
  ["path", { d: "m4.93 4.93 1.41 1.41" }],
  ["path", { d: "m17.66 17.66 1.41 1.41" }],
  ["path", { d: "M2 12h2" }],
  ["path", { d: "M20 12h2" }],
  ["path", { d: "m6.34 17.66-1.41 1.41" }],
  ["path", { d: "m19.07 4.93-1.41 1.41" }],
];
const ARROW_UP = [
  ["path", { d: "m5 12 7-7 7 7" }],
  ["path", { d: "M12 19V5" }],
];
const CHEVRONS_UP = [
  ["path", { d: "m17 11-5-5-5 5" }],
  ["path", { d: "m17 18-5-5-5 5" }],
];
const COPY = [
  ["rect", { width: "14", height: "14", x: "8", y: "8", rx: "2", ry: "2" }],
  ["path", { d: "M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" }],
];
const CHECK = [["path", { d: "M20 6 9 17l-5-5" }]];
const CHEVRON_RIGHT = [["path", { d: "m9 18 6-6-6-6" }]];
const CHEVRON_DOWN = [["path", { d: "m6 9 6 6 6-6" }]];

const SECTIONS = [
  { label: "Getting Started", fragments: ["/getting-started/"], icon: ROCKET },
  {
    label: "Using Praxist",
    fragments: [
      "/user-guide/",
      "/guides/task-projects.html",
      "/guides/examples-and-templates.html",
      "/examples/",
      "/guides/operators.html",
      "/guides/user-facing-reports-and-init.html",
      "/guides/costs.html",
    ],
    icon: TERMINAL,
  },
  {
    label: "Research System",
    fragments: [
      "/concepts/architecture.html",
      "/concepts/panel-topology-prompts.html",
      "/guides/research-loop-",
      "/guides/deep-innovation-gate.html",
      "/guides/qdig-cohort-allocator.html",
      "/guides/peer-local-structured-memory-long-context.html",
      "/guides/central-resource-scheduler.html",
      "/guides/scientific-literature-lookup.html",
    ],
    icon: FLASK,
  },
  {
    label: "Operations",
    fragments: [
      "/operations/",
      "/guides/agent-runtimes.html",
      "/guides/model-providers.html",
      "/guides/credentials.html",
      "/guides/cost-optimization.html",
      "/guides/workflow-stages.html",
      "/guides/tool-servers.html",
      "/guides/budget-policies.html",
    ],
    icon: GAUGE,
  },
  {
    label: "Developer Guide",
    fragments: [
      "/guides/contributing.html",
      "/concepts/config-discipline.html",
      "/concepts/runtime-model.html",
      "/guides/plugins.html",
      "/guides/research-topology-and-module-api.html",
    ],
    icon: CODE,
  },
  { label: "Reference", fragments: ["/reference/"], icon: BOOK_OPEN },
  { label: "About", fragments: ["/about/", "/guides/legacy-migration.html"], icon: INFO },
];

const HOME_SECTION = { label: "Praxist", icon: COMPASS };
const PAGE_ICON_STORAGE_KEY = "praxist.page-morphicon.route";
let disposers = [];

function currentSection() {
  const pagePath = window.location.pathname.toLowerCase();
  return (
    SECTIONS.find(({ fragments }) =>
      fragments.some((fragment) => pagePath.includes(fragment)),
    ) ?? HOME_SECTION
  );
}

function currentPageLabel(fallback) {
  const heading = document.querySelector(".md-content__inner > h1");
  if (!heading) return fallback;
  const copy = heading.cloneNode(true);
  copy.querySelectorAll(".headerlink").forEach((link) => link.remove());
  return copy.textContent.trim() || fallback;
}

function pageForPath(pathname, fallback = null) {
  const pagePath = pathname.toLowerCase();
  for (const [route, icon] of PAGE_ICONS) {
    if (pagePath.endsWith(route)) return { route, icon };
  }
  return fallback;
}

function currentPage(section) {
  return (
    pageForPath(window.location.pathname) ?? {
      route: section === HOME_SECTION ? "/index.html" : window.location.pathname,
      icon: section.icon,
    }
  );
}

function previousPageIcon() {
  try {
    const route = window.sessionStorage.getItem(PAGE_ICON_STORAGE_KEY);
    return route ? pageForPath(route)?.icon ?? null : null;
  } catch {
    return null;
  }
}

function rememberPage(route) {
  try {
    window.sessionStorage.setItem(PAGE_ICON_STORAGE_KEY, route);
  } catch {
    // Storage can be unavailable in privacy-restricted browser contexts.
  }
}

function iconSurface(container, initialIcon) {
  const svg = document.createElementNS(SVG_NAMESPACE, "svg");
  const path = document.createElementNS(SVG_NAMESPACE, "path");
  svg.dataset.praxistMorphicon = "";
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("focusable", "false");
  svg.append(path);

  const previous = container.querySelector(":scope > svg");
  if (previous) previous.replaceWith(svg);
  else container.prepend(svg);
  container.classList.add("praxist-morphicon-control");

  const morph = createMorph(path, initialIcon, { reducedMotion: "user" });
  return {
    morph,
    destroy() {
      morph.destroy();
    },
  };
}

function register(dispose) {
  disposers.push(dispose);
}

function clearMorphicons() {
  for (const dispose of disposers.reverse()) dispose();
  disposers = [];
}

function mountSidebarIdentity() {
  const scrollwrap = document.querySelector(
    ".md-sidebar--primary .md-sidebar__scrollwrap",
  );
  if (!scrollwrap) return;

  scrollwrap.querySelector(":scope > .praxist-sidebar-identity")?.remove();
  const section = currentSection();
  const page = currentPage(section);
  const host = document.createElement("div");
  const icon = document.createElement("span");
  const label = document.createElement("span");
  host.className = "praxist-sidebar-identity";
  icon.className = "praxist-sidebar-identity__icon";
  icon.setAttribute("aria-hidden", "true");
  label.className = "praxist-sidebar-identity__label";
  label.textContent = currentPageLabel(section.label);
  host.title = label.textContent;
  host.append(icon, label);
  scrollwrap.prepend(host);

  const initialIcon = previousPageIcon() ?? section.icon;
  const surface = iconSurface(icon, initialIcon);
  let frame = 0;
  if (initialIcon !== page.icon) {
    frame = requestAnimationFrame(() => {
      frame = requestAnimationFrame(() =>
        surface.morph.morphTo(page.icon, "smooth"),
      );
    });
  }
  rememberPage(page.route);

  const showSection = () => surface.morph.morphTo(section.icon, "snappy");
  const showPage = () => surface.morph.morphTo(page.icon, "snappy");
  host.addEventListener("pointerenter", showSection);
  host.addEventListener("pointerleave", showPage);

  register(() => {
    cancelAnimationFrame(frame);
    host.removeEventListener("pointerenter", showSection);
    host.removeEventListener("pointerleave", showPage);
    surface.destroy();
    host.remove();
  });
}

function mountToggle(input, containers, offIcon, onIcon) {
  for (const container of containers) {
    const surface = iconSurface(container, input.checked ? onIcon : offIcon);
    const sync = () =>
      surface.morph.morphTo(input.checked ? onIcon : offIcon, "snappy");
    input.addEventListener("change", sync);
    register(() => {
      input.removeEventListener("change", sync);
      surface.destroy();
    });
  }
}

function mountCheckboxControls() {
  const drawer = document.querySelector("#__drawer");
  if (drawer) {
    mountToggle(
      drawer,
      document.querySelectorAll('label.md-header__button[for="__drawer"]'),
      MENU,
      X,
    );
  }

  const search = document.querySelector("#__search");
  if (search) {
    mountToggle(
      search,
      document.querySelectorAll('label[for="__search"]:not(.md-search__overlay)'),
      SEARCH,
      X,
    );
  }

  for (const input of document.querySelectorAll("input.md-nav__toggle[id]")) {
    const selector = `label[for="${CSS.escape(input.id)}"] .md-nav__icon`;
    mountToggle(
      input,
      document.querySelectorAll(selector),
      CHEVRON_RIGHT,
      CHEVRON_DOWN,
    );
  }
}

function mountHoverControl(container, restingIcon, hoverIcon) {
  const surface = iconSurface(container, restingIcon);
  const enter = () => surface.morph.morphTo(hoverIcon, "snappy");
  const leave = () => surface.morph.morphTo(restingIcon, "snappy");
  container.addEventListener("pointerenter", enter);
  container.addEventListener("pointerleave", leave);
  register(() => {
    container.removeEventListener("pointerenter", enter);
    container.removeEventListener("pointerleave", leave);
    surface.destroy();
  });
}

function mountHoverControls() {
  for (const label of document.querySelectorAll(
    'label.md-header__button[for="__palette_1"]',
  )) {
    mountHoverControl(label, MOON, SUN);
  }
  for (const label of document.querySelectorAll(
    'label.md-header__button[for="__palette_0"]',
  )) {
    mountHoverControl(label, SUN, MOON);
  }
  for (const button of document.querySelectorAll("button.md-top")) {
    mountHoverControl(button, ARROW_UP, CHEVRONS_UP);
  }
}

function mountFeedbackControl(container, restingIcon, feedbackIcon, delay = 900) {
  if (container.dataset.praxistMorphiconMounted === "true") return;
  container.dataset.praxistMorphiconMounted = "true";
  const surface = iconSurface(container, restingIcon);
  let timer = 0;
  const feedback = () => {
    window.clearTimeout(timer);
    surface.morph.morphTo(feedbackIcon, "snappy");
    timer = window.setTimeout(
      () => surface.morph.morphTo(restingIcon, "smooth"),
      delay,
    );
  };
  container.addEventListener("click", feedback);
  register(() => {
    window.clearTimeout(timer);
    container.removeEventListener("click", feedback);
    delete container.dataset.praxistMorphiconMounted;
    surface.destroy();
  });
}

function mountFeedbackControls() {
  const scan = () => {
    for (const button of document.querySelectorAll(
      "button.md-code__button, button.md-clipboard",
    )) {
      mountFeedbackControl(button, COPY, CHECK, 1200);
    }
  };
  scan();

  const observer = new MutationObserver(scan);
  const content = document.querySelector(".md-content") ?? document.body;
  observer.observe(content, { childList: true, subtree: true });
  register(() => observer.disconnect());

  for (const button of document.querySelectorAll(
    '.md-search__options button[type="reset"]',
  )) {
    mountFeedbackControl(button, X, CHECK, 600);
  }
}

function mountMorphicons() {
  clearMorphicons();
  mountSidebarIdentity();
  mountCheckboxControls();
  mountHoverControls();
  mountFeedbackControls();
}

if (typeof document$ !== "undefined") {
  document$.subscribe(mountMorphicons);
} else if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", mountMorphicons, { once: true });
} else {
  mountMorphicons();
}
