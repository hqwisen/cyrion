<template>
  <div class="container">
    <div class="row justify-content-center top-buffer">
      <div class="col-12">
        <div class="btn-group">
          <button @click="addCanvas" class="btn btn-info">Add canvas</button>
          <button @click="clearAllCanvas" class="btn btn-danger">Clear all</button>
          <button @click="compute" class="btn btn-success">Compute</button>
        </div>
      </div>
    </div>
    <div class="row top-buffer">

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
      numberOfCanvas: 1,
      apiUrl: 'http://localhost:8000/'
    }
  },
  methods: {
    addCanvas: function () {
      this.numberOfCanvas++;
    },
    clearAllCanvas: function () {

      this.$refs.canvasComponents.forEach(function (canvas) {
        canvas.clear()
      })
    },
    compute: function () {
      let allDataImg = [];
      this.$refs.canvasComponents.forEach(function (canvas) {
        allDataImg.push(canvas.toDataURL())
      });
      this.computeResult(allDataImg);
    },
    computeResult: function (allDataImg) {
      console.log("Sending request to", this.apiUrl, "api/basic/upload");
      let data = {
        'samples': allDataImg
      };
      this.$http.post(this.apiUrl + 'api/basic/upload', data).then(response => {
        console.log("Response success with status:", response.status)
      }, response => {
        console.log("Response FAILURE:", response.status)
      });
    }
  }
}
</script>

<style scoped>

</style>
