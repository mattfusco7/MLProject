$("#url_radio").on("change", function () {

  $(".url_label").removeClass("hidden");
  $("#url").removeClass("hidden");
  $("#url").attr("required", true);
  $(".file_label").addClass("hidden");
  $("#file").addClass("hidden");
  $("#file").attr("required", false);
})

$("#file_radio").on("change", function () {

  $(".file_label").removeClass("hidden");
  $("#file").removeClass("hidden");
  $("#file").attr("required", true);
  $(".url_label").addClass("hidden");
  $("#url").addClass("hidden");
  $("#url").attr("required", false);
})

$("#bottom_link").on("click", function () {

  $(".matches_label").removeClass("hidden");
  $("#matches").removeClass("hidden");
  $(".btn-matches").removeClass("hidden");
})


$(".btn-matches").on("click", function () {

  $(".matches_label").addClass("hidden");
  $("#matches").addClass("hidden");
  $(".btn-matches").addClass("hidden");
})