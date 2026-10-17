// Shared helpers for the figure pages.
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

// draw only after the web font is loaded, so text measurements (legend spacing, label placement) are exact
function whenFonts(draw) {
  const fonts = ["400 12px SS3", "600 12px SS3", "700 12px SS3", "italic 400 12px SS3",
                 "12px KaTeX_Main", "italic 12px KaTeX_Math", "bold 12px KaTeX_Main", "12px KaTeX_Size1"];
  Promise.all(fonts.map(f => document.fonts.load(f))).then(() => { draw(); window.__READY__ = true; });
}

function makeSvg(W, H, label) {
  const svg = d3.select("#fig").append("svg").attr("width", W).attr("height", H).attr("viewBox", `0 0 ${W} ${H}`)
    .attr("role", "img").attr("aria-label", label);
  svg.append("defs").html(`
    <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="${css('--ink-2')}"/></marker>
    <marker id="arrB" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="${css('--calm')}"/></marker>`);
  return svg;
}

function panelTitle(g, x, y, letter, text) {
  g.append("text").attr("class", "t-panel").attr("x", x).attr("y", y).text(letter);
  if (text) g.append("text").attr("class", "t-title").attr("x", x + 13).attr("y", y).text(text);
}

// text with simple math: parts = [["κ"], ["h", "sub"], [" ≈ h"]]  (sub/sup shrink and shift)
function richText(g, x, y, parts, cls = "t-label", attrs = {}) {
  const t = g.append("text").attr("class", cls).attr("x", x).attr("y", y);
  Object.entries(attrs).forEach(([k, v]) => t.attr(k, v));
  parts.forEach(([s, kind, extra]) => {
    const sp = t.append("tspan").text(s);
    if (kind === "sub") sp.attr("dy", "0.32em").attr("font-size", "0.74em");
    if (kind === "sup") sp.attr("dy", "-0.45em").attr("font-size", "0.74em");
    if (kind === "back-sub") sp.attr("dy", "-0.32em");
    if (kind === "back-sup") sp.attr("dy", "0.45em");
    if (kind === "it") sp.attr("font-style", "italic");
    if (extra) Object.entries(extra).forEach(([k, v]) => sp.attr(k, v));
  });
  return t;
}

// KaTeX math placed over the SVG (absolutely positioned HTML; prints as vector text).
// (x, y) = start (or middle/end with opts.anchor) of the math baseline, in figure px.
function mathAt(x, y, tex, opts = {}) {
  const fig = document.getElementById("fig");
  fig.style.position = "relative";
  const d = document.createElement("div");
  d.style.position = "absolute";
  d.style.whiteSpace = "nowrap";
  d.style.fontSize = (opts.size || 12) + "px";
  d.style.color = opts.color || "#0b0b0b";
  d.style.left = "0px"; d.style.top = "0px";
  katex.render(tex, d, { throwOnError: true, displayMode: false });
  const mark = document.createElement("span");
  mark.style.display = "inline-block"; mark.style.width = "0"; mark.style.height = "0"; mark.style.verticalAlign = "baseline";
  d.appendChild(mark);
  fig.appendChild(d);
  const fr = fig.getBoundingClientRect(), dr = d.getBoundingClientRect(), mr = mark.getBoundingClientRect();
  const baseOff = mr.top - dr.top;
  const w = dr.width;
  let left = x;
  if (opts.anchor === "middle") left = x - w / 2;
  if (opts.anchor === "end") left = x - w;
  d.style.left = left + "px";
  d.style.top = (y - baseOff) + "px";
  return { node: d, width: w };
}
