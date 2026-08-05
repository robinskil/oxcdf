/* Generates `szip_test.h5`, a corpus file for the szip filter.
 *
 * This reader does not decode szip yet. The filter is Rice coding (CCSDS
 * 121.0), and writing an entropy decoder that nothing validates would be the
 * one part of this crate not checked against a second implementation.
 *
 * This fixture removes that blocker: the local HDF5 has both the szip encoder
 * and decoder, so once a decoder exists it can be compared against netcdf-c the
 * same way every other filter is. Until then `tests/new_features.rs` uses this
 * file to prove the fallback contract: an szip dataset is listed, reports
 * `is_readable() == false`, and fails loudly rather than returning wrong bytes.
 *
 * Rebuild with:
 *   h5cc -o generate_szip generate_szip.c && ./generate_szip
 */
#include "hdf5.h"
#include <stdio.h>
#define NX 64
#define NY 8
int main(void){
  hid_t f=H5Fcreate("szip_test.h5",H5F_ACC_TRUNC,H5P_DEFAULT,H5P_DEFAULT);
  hsize_t dims[2]={NX,NY}, ch[2]={16,8};
  hid_t sp=H5Screate_simple(2,dims,NULL);
  hid_t dcpl=H5Pcreate(H5P_DATASET_CREATE);
  H5Pset_chunk(dcpl,2,ch);
  herr_t e=H5Pset_szip(dcpl,H5_SZIP_NN_OPTION_MASK,8);
  printf("set_szip=%d\n",(int)e);
  if(e<0){ printf("SZIP ENCODER UNAVAILABLE\n"); return 1; }
  hid_t d=H5Dcreate2(f,"szipped",H5T_STD_I32LE,sp,H5P_DEFAULT,dcpl,H5P_DEFAULT);
  int buf[NX*NY]; for(int i=0;i<NX*NY;i++) buf[i]=i*7-500;
  e=H5Dwrite(d,H5T_NATIVE_INT,H5S_ALL,H5S_ALL,H5P_DEFAULT,buf);
  printf("write=%d\n",(int)e);
  H5Dclose(d);H5Pclose(dcpl);H5Sclose(sp);H5Fclose(f);
  return 0;}
