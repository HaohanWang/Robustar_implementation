import Vue from 'vue';
import VueRouter from 'vue-router';
import Home from '../views/Home.vue';
import ImageList from '../views/ImageList.vue';
import TrainPad from '../views/TrainPad.vue';
import InfluencePad from '../views/InfluencePad.vue';
import EditImage from '@/views/EditImage';
import AutoAnnotatePad from '@/views/AutoAnnotatePad';
import Prediction from '@/views/Prediction.vue';
import TestPad from '@/views/TestPad';
import Config from '@/views/Config';

Vue.use(VueRouter);

const routes = [
  {
    path: '/',
    name: 'Home',
    component: Home,
  },
  {
    path: '/about',
    name: 'About',
    // route level code-splitting
    // this generates a separate chunk (about.[hash].js) for this route
    // which is lazy-loaded when the route is visited.
    component: () => import(/* webpackChunkName: "about" */ '../views/About.vue'),
  },
  {
    path: '/image-list/:split',
    name: 'ImageList',
    component: ImageList,
  },
  {
    path: '/train-pad',
    name: 'TrainPad',
    component: TrainPad,
  },
  {
    path: '/edit/:mode',
    name: 'EditImage',
    component: EditImage,
    props: true,
  },
  {
    path: '/test',
    name: 'TestPad',
    component: TestPad,
  },
  {
    path: '/influence-pad',
    name: 'InfluencePad',
    component: InfluencePad,
  },
  {
    path: '/predict',
    name: 'Prediction',
    component: Prediction,
  },
  {
    path: '/config',
    name: 'Config',
    component: Config,
  },
  {
    path: '/auto-annotate',
    name: 'AutoAnnotatePad',
    component: AutoAnnotatePad,
  },
];

const router = new VueRouter({
  routes,
});

export default router;
