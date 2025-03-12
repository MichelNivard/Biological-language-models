# install biomartr 1.0.7 from CRAN
install.packages("biomartr", dependencies = TRUE)
# Install Biostrings if not installed
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}
# Load required package
library(Biostrings)
library(biomartr)


# download the genome of Homo sapiens from ensembl
# and store the corresponding genome CDS file in '_ncbi_downloads/CDS'
HS.cds.ensembl <- getCDS( db       = "ensembl", 
                         organism = "Homo sapiens",
                         path     = file.path("_ncbi_downloads","CDS"))

# import downloaded CDS as Biostrings object
Human_CDS <- read_cds(file     = HS.cds.ensembl, 
  obj.type = "Biostrings")

# Extract headers and sequences
headers <- names(Human_CDS )
sequences <- as.character(Human_CDS )

# Extract metadata using regex
extract_metadata <- function(header) {
  # Extract key-value pairs
  transcript_id <- sub("^>([^ ]+).*", "\\1", header)
  chromosome <- sub(".*chromosome:([^ ]+).*", "\\1", header)
  start <- sub(".*chromosome:[^:]+:([^:]+).*", "\\1", header)
  end <- sub(".*chromosome:[^:]+:[^:]+:([^:]+).*", "\\1", header)
  strand <- sub(".*chromosome:[^:]+:[^:]+:[^:]+:([^ ]+).*", "\\1", header)
  gene_id <- sub(".*gene:([^ ]+).*", "\\1", header)
  gene_biotype <- sub(".*gene_biotype:([^ ]+).*", "\\1", header)
  transcript_biotype <- sub(".*transcript_biotype:([^ ]+).*", "\\1", header)
  gene_symbol <- sub(".*gene_symbol:([^ ]+).*", "\\1", header)
  
  # Extract everything after "description:"
  description <- sub(".*description:(.*)", "\\1", header)
  
  # Return as a list
  list(
      transcript_id = transcript_id,
      chromosome = chromosome,
      start = start,
      end = end,
      strand = strand,
      gene_id = gene_id,
      gene_biotype = gene_biotype,
      transcript_biotype = transcript_biotype,
      gene_symbol = gene_symbol,
      description = description
  )
}

# Apply metadata extraction
metadata_list <- lapply(headers, extract_metadata)

# Convert to a data frame
metadata_df <- do.call(rbind, lapply(metadata_list, as.data.frame))
metadata_df$sequence <- sequences  # Add sequences to the data frame

# Save as CSV
write.csv(metadata_df, "genome_sequences.csv", row.names = FALSE, quote = TRUE)

# Print first few rows to check
head(metadata_df)
