import Vue from 'vue'
import Router from 'vue-router'
import BasicRecognition from '@/components/BasicRecognition'
import VueResource from 'vue-resource'

Vue.use(Router)
Vue.use(VueResource)



export default new Router({
  routes: [
    {
      path: '/',
      name: 'Basic Recognition',
      component: BasicRecognition
    }
  ]
})
