<template>
  <div class="col-3">
  <canvas @mousedown="mouseDown" @mouseup="mouseUp" @mouseout="mouseOut" @mousemove="mouseMove"
          ref="canvas" :width="width" :height="height"></canvas>
    <span v-if="prediction">Prediction: <b>{{ prediction }}</b></span>
  </div>
</template>

<script>
export default {
  name: 'drawing-canvas',
  data() {
    return {
      prevX: 0,
      currX: 0,
      prevY: 0,
      currY: 0,
      ctx: null,
      canvas: null,
      drawing: false,
      startDrawing: false,
      lineWidth: 11,
      width: 280,
      height: 280,
      prediction: ""
    }
  },
  mounted: function () {
    this.init()
  },
  methods: {
    init: function () {
      this.canvas = this.$refs.canvas;
      this.ctx = this.canvas.getContext("2d");
      // Avoid transparent background
      this.ctx.fillStyle = "white";
      this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    },
    mouseDown: function (e) {
      this.setCurrentCoord(e);
      this.drawing = true;
      this.startDrawing = true;
      if (this.startDrawing) {
        this.ctx.beginPath();
        this.ctx.fillRect(this.currX, this.currY, this.lineWidth, this.lineWidth);
        this.ctx.closePath();
        this.startDrawing = false;
      }

    },
    mouseUp: function () {
      this.drawing = false
    },
    mouseOut: function () {
      this.drawing = false
    },
    mouseMove: function (e) {
      if (this.drawing) {
        this.setCurrentCoord(e)
        this.draw()
      }
    },
    draw: function () {
      if (this.ctx != null) {
        this.ctx.beginPath()
        this.ctx.moveTo(this.prevX, this.prevY)
        this.ctx.lineTo(this.currX, this.currY)
        this.ctx.lineWidth = this.lineWidth
        this.ctx.stroke()
        this.ctx.closePath()
      } else {
        console.log("Can't draw, context not set")
      }
    },
    setCurrentCoord: function (e) {
      this.prevX = this.currX;
      this.prevY = this.currY;
      var rect = this.canvas.getBoundingClientRect();
      this.currX = e.clientX - rect.left;
      this.currY = e.clientY - rect.top;
    },
    clear: function () {
      this.ctx.fillRect(0, 0, this.width, this.height);
    },
    toDataURL: function(){
      return this.canvas.toDataURL('image/png')
    }
  }
}
/*
  function init() {
    w = canvas.width;
    h = canvas.height;
  }

  function erase() {
    var m = confirm("Want to clear");
    if (m) {
      ctx.clearRect(0, 0, w, h);
      document.getElementById("canvasimg").style.display = "none";
    }
  }

  function save() {
    document.getElementById("canvasimg").style.border = "2px solid";
    var dataURL = canvas.toDataURL();
    document.getElementById("canvasimg").src = dataURL;
    document.getElementById("canvasimg").style.display = "inline";
  }

*/
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
canvas {
  background: #FFF;
  border: 1px solid black;
}

h1, h2 {
  font-weight: normal;
}

ul {
  list-style-type: none;
  padding: 0;
}

li {
  display: inline-block;
  margin: 0 10px;
}

a {
  color: #42b983;
}
</style>

