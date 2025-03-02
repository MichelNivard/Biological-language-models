# install biomartr 1.0.7 from CRAN
#install.packages("biomartr", dependencies = TRUE)
# Install Biostrings if not installed
#if (!requireNamespace("BiocManager", quietly = TRUE)) {
#  install.packages("BiocManager")
#}
# Load required package
library(Biostrings)
library(biomartr)
ak <- listGenomes(db = "ensembl",type = "all",subset = "EnsemblVertebrates")

# specify the species names
download_species <- c("Homo sapiens",
                      "Vicugna pacos",
                      "papio anubis",
                      "Mus musculus", 
                      "Cavia porcellus",
                    "Canis lupus familiarisgsd",
                    "Pan paniscus",
                    "Panthera leo",
                    "Otus sunia",
                    "Falco tinnunculus",
                    "Apteryx haastii",
                  "Electrophorus electricus",
                  "Takifugu rubripes")

# retrieve these three species from NCBI RefSeq                       
getCDSSet("ensembl", organisms = download_species, path = "set_of_cds")

# import downloaded CDS as Biostrings object
cds_path <- "set_of_cds"

# List all FASTA files in the directory
fasta_files <- list.files(path = cds_path, pattern = "\\.fa$", full.names = TRUE)

# Read all CDS files and merge into one Biostrings object
Cross_species_CDS <- do.call(c, lapply(fasta_files, readDNAStringSet))

# Check summary
print(Cross_species_CDS)

# Extract headers and sequences
headers <- names(Cross_species_CDS)
sequences <- as.character(Cross_species_CDS )

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
write.csv(metadata_df, "genome_sequences_.csv", row.names = FALSE, quote = TRUE)

# Print first few rows to check
head(metadata_df)
