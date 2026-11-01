// Toggle the version/language switcher flyout.
document.addEventListener("DOMContentLoaded", function () {
  var flyout = document.getElementById("versions-flyout");
  if (!flyout) {
    return;
  }
  var toggle = flyout.querySelector(".rst-current-version");
  toggle.addEventListener("click", function (event) {
    event.stopPropagation();
    flyout.classList.toggle("rst-open");
  });
  document.addEventListener("click", function (event) {
    if (flyout.classList.contains("rst-open") && !flyout.contains(event.target)) {
      flyout.classList.remove("rst-open");
    }
  });
});
