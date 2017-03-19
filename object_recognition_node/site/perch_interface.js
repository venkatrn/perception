function perch_interface()
{
  var self = this;

  // Setup the list of objects. We'll assume there is an object_name.png
  // file under images.
  this.objects = [
    "003_cracker_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "010_potted_meat_can",
    "011_banana",
    "019_pitcher_base",
    "024_bowl",
    "025_mug",
    "035_power_drill",
    "052_extra_large_clamp"
      ];
  this._setup_ros();
}

perch_interface.prototype._setup_ros = function ()
{
  var self = this;

  this.ros = new ROSLIB.Ros({
    url : 'ws://' + location.hostname + ':9090'
  });

  this.ros.on('connection', function() {
    console.log('Connected to websocket server.');
  });

  this.ros.on('error', function(error) {
    console.log('Error connecting to websocket server: ', error);
  });

  this.ros.on('close', function() {
    console.log('Connection to websocket server closed.');
  });

  /////////////////////////
  // Set up publishers //
  /////////////////////////

  this.command_pub_ = new ROSLIB.Topic({
    ros: this.ros,
    name: 'requested_object',
    messageType: 'std_msgs/String'
  });
}

perch_interface.prototype.send_object_name = function (div)
{
  this.reset_div_borders();
  var img = div.childNodes[0]
    object_id = img.getAttribute("id");
  // div.style.outline = '5px solid orange';
  div.setAttribute('style', 'box-shadow: 0px 0px 30px 5px green');

  var message = new ROSLIB.Message({
    data: object_id
  });
  console.log('Sending object ' + message.data + '!');
  this.command_pub_.publish(message);
}

perch_interface.prototype.reset_div_borders = function ()
{
  var divs = document.getElementsByClassName("boxInner");
  for(var i = 0; i < divs.length ; i++) {
    // divs[i].style.outline = 'none';
    divs[i].setAttribute('style', ':inactive {box-shadow: none}');
  }
}

var iface = new perch_interface();
var objects = iface.objects;
for(var i = 0; i < objects.length ; i++) {
  console.log("Adding object " + objects[i]);

  var outer_box_div = document.createElement("div");
  outer_box_div.setAttribute("class", "box");

  var box_div = document.createElement("div");
  box_div.setAttribute("class", "boxInner");

  var img = document.createElement("img");
  img.setAttribute("src", "images/" + objects[i] + ".png");
  img.setAttribute("id", objects[i]);

  var inner_box_div = document.createElement("div");
  inner_box_div.setAttribute("class", "titleBox");
  inner_box_div.innerHTML = objects[i];

  box_div.appendChild(img);
  box_div.appendChild(inner_box_div);
  outer_box_div.appendChild(box_div);
  document.getElementById("wrap_id").appendChild(outer_box_div);
}

var all_divs = document.getElementsByClassName('boxInner');
for(var i = 0; i < all_divs.length ; i++) {
  div = all_divs[i];
  div.addEventListener("click", function() {iface.send_object_name(this)}, false);
  // img.addEventListener("click", function() {this.style.outline = '5px solid orange'});
}
