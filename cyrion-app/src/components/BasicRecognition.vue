<template>
  <div class="container">
    <div class="col-12">
      <drawing-canvas></drawing-canvas>
    </div>
  </div>
</template>

<script>
import DrawingCanvas from "./Canvas";

export default {
  name: "BasicRecognition",
  components: {DrawingCanvas},
  data() {
    return {
      numberOfCanvas: 1
    }
  }
}
</script>

<style scoped>

</style>
