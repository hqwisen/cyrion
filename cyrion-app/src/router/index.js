import Vue from 'vue'
import Router from 'vue-router'
import BasicRecognition from '@/components/BasicRecognition'

Vue.use(Router)

export default new Router({
  routes: [
    {
      path: '/',
      name: 'Basic Recognition',
      component: BasicRecognition
    }
  ]
})
