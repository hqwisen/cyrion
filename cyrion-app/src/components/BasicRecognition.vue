<template>
  <div class="container">
    <div class="row">
      <div class="col">
        <div class="btn-group">
          <button @click="addCanvas" class="btn btn-info">Add canvas</button>
          <button @click="clearAllCanvas" class="btn btn-danger">Clear all</button>
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
    }
  }
}
</script>

<style scoped>

</style>
