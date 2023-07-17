#ifndef EXTERNAL_RETRIEVAL_TESTUTILS_H
#define EXTERNAL_RETRIEVAL_TESTUTILS_H

#include "usearch.h"
#include <stdlib.h>

void prepare_external_index(char* mapped_index, int dim, size_t num_vectors, usearch_metadata_t* metadata);
void free_external_index();
void* node_retriever(int id);
void* node_retriever_mut(int id);
#endif