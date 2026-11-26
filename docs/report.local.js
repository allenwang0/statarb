/* statarb report: charts drawn from the inlined #sa-data JSON with report-kit v1 helpers. */
(function () {
  "use strict";
  var D = RK.data("sa-data");
  if (!D || !D.real) return;
  var R = D.real, Y = R.pair[0], X = R.pair[1];
  var MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  var ym = function (t) { var d = new Date(t * 1000); return MONTHS[d.getUTCMonth()] + " " + d.getUTCFullYear(); };
  var iso = function (t) { return new Date(t * 1000).toISOString().slice(0, 10); };
  var sgn = function (v, n) { return (v < 0 ? "−" : "") + Math.abs(v).toFixed(n); };
  var KAL = "var(--series-1)", OLS = "var(--series-2)";

  /* Test-window cumulative return */
  var T = R.test;
  RKCharts.line("#c-test", {
    title: "Cumulative return on the test window, Kalman vs rolling OLS",
    x: T.t, xFormat: ym, format: function (v) { return sgn(v, 0) + " %"; }, yUnit: "%",
    series: [{ key: "k", name: "Kalman", color: KAL, y: T.kalman }, { key: "o", name: "Rolling OLS", color: OLS, y: T.ols, dash: true }],
    yLabel: "Cumulative return (%)", xLabel: "Date", refs: [{ y: 0, label: "" }]
  });
  var monthly = [];
  T.t.forEach(function (t, i) { var next = T.t[i + 1]; if (next === undefined || iso(next).slice(0, 7) !== iso(t).slice(0, 7)) monthly.push({ d: iso(t), k: sgn(T.kalman[i], 1), o: sgn(T.ols[i], 1) }); });
  RKCharts.table("#t-test", { caption: "Cumulative return (%) at each month's last trading day",
    columns: [{ key: "d", label: "Date" }, { key: "k", label: "Kalman" }, { key: "o", label: "Rolling OLS" }], rows: monthly });

  /* Hedge ratio */
  var B = R.beta;
  RKCharts.line("#c-beta", {
    title: "Hedge ratio, Kalman with ±2 sd band vs rolling OLS", x: B.t, xFormat: ym, zero: false,
    format: function (v) { return v.toFixed(2); },
    series: [{ key: "k", name: "Kalman", color: KAL, y: B.kalman, lo: B.lo, hi: B.hi }, { key: "o", name: "Rolling OLS", color: OLS, y: B.ols }],
    yLabel: "Hedge ratio (" + X + " shares per " + Y + " share)", xLabel: "Date"
  });
  RKCharts.table("#t-beta", { caption: "Hedge ratio, every 5th trading day",
    columns: [{ key: "d", label: "Date" }, { key: "k", label: "Kalman" }, { key: "lo", label: "Kalman −2 sd" }, { key: "hi", label: "Kalman +2 sd" }, { key: "o", label: "Rolling OLS" }],
    rows: B.t.map(function (t, i) { return { d: iso(t), k: B.kalman[i].toFixed(3), lo: B.lo[i].toFixed(3), hi: B.hi[i].toFixed(3), o: B.ols[i].toFixed(3) }; }) });

  /* Autocorrelation of standardized errors */
  var A = R.acf;
  RKCharts.line("#c-acf", {
    title: "Autocorrelation of the Kalman model's standardized errors by lag", x: A.lag, xFormat: function (v) { return v + ""; },
    format: function (v) { return v.toFixed(1); }, endLabels: false,
    ymin: Math.min(-A.band, Math.min.apply(null, A.v)),
    series: [{ key: "a", name: "Autocorrelation", color: KAL, y: A.v }],
    yLabel: "Autocorrelation", xLabel: "Lag (trading days)",
    refs: [{ y: A.band, label: "95 % band" }, { y: -A.band, label: "" }]
  });
  RKCharts.table("#t-acf", { caption: "Autocorrelation by lag; white-noise 95 % band ±" + A.band.toFixed(3),
    columns: [{ key: "l", label: "Lag (days)" }, { key: "v", label: "Autocorrelation" }, { key: "o", label: "Outside band" }],
    rows: A.lag.map(function (l, i) { return { l: l, v: A.v[i].toFixed(3), o: Math.abs(A.v[i]) > A.band ? "yes" : "no" }; }) });
})();
