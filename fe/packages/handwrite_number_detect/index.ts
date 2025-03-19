/**
 * @file index.ts
 * @author afcfzf(9301462@qq.com)
 */

import tf from '@tensorflow/tfjs-node';
import mnist from 'mnist';

const loadData = () => {
  const set = mnist.set(8000, 2000);
  const training = set.training;
  const test = set.test;

  const formData = (data) => {
    return {
      images: tf.tensor(data.map())
    };
  };


};


