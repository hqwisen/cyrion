<template>
  <div class="container">
    <div class="row">
      <div class="col">
        <div class="btn-group">
          <button @click="addCanvas" class="btn btn-info">Add canvas</button>
          <button @click="clearAllCanvas" class="btn btn-danger">Clear all</button>
          <button @click="compute" class="btn btn-success">Compute</button>
        </div>
      </div>
    </div>
    <div class="row">

      <div class="col-12">
        <drawing-canvas v-for="i in numberOfCanvas" :key="i"
                        ref="canvasComponents"></drawing-canvas>
      </div>
    </div>
  </div>
</template>

<script>
import DrawingCanvas from "./DrawingCanvas";

export default {
  name: "BasicRecognition",
  components: {DrawingCanvas},
  data() {
    return {
      numberOfCanvas: 5
    }
  },
  methods: {
    addCanvas: function () {
      this.numberOfCanvas++
    },
    clearAllCanvas: function () {

      this.$refs.canvasComponents.forEach(function (canvas) {
        canvas.clear()
      })
    },
    compute: function () {
      let allDataImg = []
      this.$refs.canvasComponents.forEach(function (canvas) {
        allDataImg.push(canvas.toDataURL())
      })
      this.computeResult(allDataImg)
    },
    computeResult: function (allDataImg) {
      console.log("Sending request to localhost:8000/api/basic/upload")
      this.$http.get('http://localhost:8000/api/basic/upload').then(response => {
        console.log("Received!")
        console.log(response)
      }, response => {
        console.log("Not received")
        console.log(response.status)
        // TODO implement error callback
      });
    }
  }
}
</script>

<style scoped>

</style>
