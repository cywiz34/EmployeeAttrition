$(".sat").change(function() {
  var $this = $(this);
  var val = $this.val();
  var span = $(".error0");
  if (val > 1 || val < 0) {
    
    span.text("value must be between 0 and 1");
    $(".sat").val("0.0");
  }
  else
  {
   span.text("");
  }
});
$(".eval").change(function() {
  var $this = $(this);
  var val = $this.val();
  var span = $(".error1");
  if (val > 1 || val < 0) {
    
    span.text("value must be between 0 and 1");
    $(".eval").val("0.0");
  }
  else
  {
   span.text("");
  }
});
$(".pc").change(function() {
  var $this = $(this);
  var val = $this.val();
  var span = $(".error2");
  if (val < 1 || val >10) {
    
    span.text("value must be between 1 and 10");
    $(".pc").val("1");
  }
  else
  {
   span.text("");
  }
});
$(".avg").change(function() {
  var $this = $(this);
  var val = $this.val();
  var span = $(".error3");
  if ( val < 50 || val >360) {
    
    span.text("value must be between 50 and 360");
    $(".avg").val("50");
  }
  else
  {
   span.text("");
  }
});
$(".years").change(function() {
  var $this = $(this);
  var val = $this.val();
  var span = $(".error4");
  if (val < 1 || val >25 ) {
    
    span.text("value must be between 1 and 25");
    $(".years").val("1");
  }
  else
  {
   span.text("");
  }
});

function wait(ms)
{
var d = new Date();
var d2 = null;
do { d2 = new Date(); }
while(d2-d < ms);
}