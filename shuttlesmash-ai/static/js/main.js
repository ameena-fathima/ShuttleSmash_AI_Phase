// static/js/main.js
// ------------------
// Shared client-side behavior for ShuttleSmash AI.
// Phase 1 has no JS-driven interactivity yet (upload progress bars and
// chart rendering are added in later phases) -- this file exists so
// base.html has a stable script include point from the start.

document.addEventListener("DOMContentLoaded", () => {
  // Auto-dismiss flash message alerts after 5 seconds.
  const alerts = document.querySelectorAll(".alert");
  alerts.forEach((alert) => {
    setTimeout(() => {
      const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
      bsAlert.close();
    }, 5000);
  });
});
