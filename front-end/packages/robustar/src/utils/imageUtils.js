import { configs } from '@/configs.js';

/**
 * Converts the index of the image within a page to its global id
 * @param {number} page_idx index of the page
 * @param {number} index index of the image within the page
 */
export const imagePageIdx2Id = (page_idx, index) => {
  return page_idx * configs.imagePerPage + index;
};

/**
 * Converts the coordinate of the image to its global id
 * @param {number} row
 * @param {number} col
 */
export const imageCoord2Idx = (row, col) => {
  return row * configs.imageListCol + col;
};

export const getPageNumber = (imageIdx) => {
  return Math.floor(imageIdx / configs.imagePerPage);
};


/*
 * Return image_url from a full url, e.g. 'http://localhost:8080/dataset/Robustar2/dataset/train/bird/115.JPEG'
 * gives 'Robustar2/dataset/train/bird/115.JPEG'
 * 
 * TODO: this feels hard coded. Does it work on all systems?
 */
export const getImageUrlFromFullUrl = (full_url) => {
  return "/" + full_url.split('/').slice(4).join("/")
};
