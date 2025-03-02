DRD2 <- "ATGGATCCACTGAATCTGTCCTGGTATGATGATGATCTGGAGAGGCAGAACTGGAGCCGGCCCTTCAACGGGTCAGACGGGAAGGCGGACAGACCCCACTACAACTACTATGCCACACTGCTCACCCTGCTCATCGCTGTCATCGTCTTCGGCAACGTGCTGGTGTGCATGGCTGTGTCCCGCGAGAAGGCGCTGCAGACCACCACCAACTACCTGATCGTCAGCCTCGCAGTGGCCGACCTCCTCGTCGCCACACTGGTCATGCCCTGGGTTGTCTACCTGGAGGTGGTAGGTGAGTGGAAATTCAGCAGGATTCACTGTGACATCTTCGTCACTCTGGACGTCATGATGTGCACGGCGAGCATCCTGAACTTGTGTGCCATCAGCATCGACAGGTACACAGCTGTGGCCATGCCCATGCTGTACAATACGCGCTACAGCTCCAAGCGCCGGGTCACCGTCATGATCTCCATCGTCTGGGTCCTGTCCTTCACCATCTCCTGCCCACTCCTCTTCGGACTCAATAACGCAGACCAGAACGAGTGCATCATTGCCAACCCGGCCTTCGTGGTCTACTCCTCCATCGTCTCCTTCTACGTGCCCTTCATTGTCACCCTGCTGGTCTACATCAAGATCTACATTGTCCTCCGCAGACGCCGCAAGCGAGTCAACACCAAACGCAGCAGCCGAGCTTTCAGGGCCCACCTGAGGGCTCCACTAAAGGAGGCTGCCCGGCGAGCCCAGGAGCTGGAGATGGAGATGCTCTCCAGCACCAGCCCACCCGAGAGGACCCGGTACAGCCCCATCCCACCCAGCCACCACCAGCTGACTCTCCCCGACCCGTCCCACCATGGTCTCCACAGCACTCCCGACAGCCCCGCCAAACCAGAGAAGAATGGGCATGCCAAAGACCACCCCAAGATTGCCAAGATCTTTGAGATCCAGACCATGCCCAATGGCAAAACCCGGACCTCCCTCAAGACCATGAGCCGTAGGAAGCTCTCCCAGCAGAAGGAGAAGAAAGCCACTCAGATGCTCGCCATTGTTCTCGGCGTGTTCATCATCTGCTGGCTGCCCTTCTTCATCACACACATCCTGAACATACACTGTGACTGCAACATCCCGCCTGTCCTGTACAGCGCCTTCACGTGGCTGGGCTATGTCAACAGCGCCGTGAACCCCATCATCTACACCACCTTCAACATTGAGTTCCGCAAGGCCTTCCTGAAGATCCTCCACTGCTGA"
nchar(DRD2)/3

# Genetic code table (Standard Code)
genetic_code <- c(
  "TTT"="F", "TTC"="F", "TTA"="L", "TTG"="L",
  "TCT"="S", "TCC"="S", "TCA"="S", "TCG"="S",
  "TAT"="Y", "TAC"="Y", "TAA"="Stop", "TAG"="Stop",
  "TGT"="C", "TGC"="C", "TGA"="Stop", "TGG"="W",
  "CTT"="L", "CTC"="L", "CTA"="L", "CTG"="L",
  "CCT"="P", "CCC"="P", "CCA"="P", "CCG"="P",
  "CAT"="H", "CAC"="H", "CAA"="Q", "CAG"="Q",
  "CGT"="R", "CGC"="R", "CGA"="R", "CGG"="R",
  "ATT"="I", "ATC"="I", "ATA"="I", "ATG"="M",
  "ACT"="T", "ACC"="T", "ACA"="T", "ACG"="T",
  "AAT"="N", "AAC"="N", "AAA"="K", "AAG"="K",
  "AGT"="S", "AGC"="S", "AGA"="R", "AGG"="R",
  "GTT"="V", "GTC"="V", "GTA"="V", "GTG"="V",
  "GCT"="A", "GCC"="A", "GCA"="A", "GCG"="A",
  "GAT"="D", "GAC"="D", "GAA"="E", "GAG"="E",
  "GGT"="G", "GGC"="G", "GGA"="G", "GGG"="G"
)

# Function to get all mutations for a codon
mutate_codon <- function(codon, codon_index, full_sequence) {
  nucleotides <- c("A", "T", "C", "G")
  mutations <- data.frame()
  
  original_aa <- genetic_code[[codon]]
  
  for (pos in 1:3) {
      original_base <- substr(codon, pos, pos)
      for (nuc in nucleotides) {
          if (nuc != original_base) {
              # Mutate the codon at this position
              mutated_codon <- codon
              substr(mutated_codon, pos, pos) <- nuc
              mutated_aa <- genetic_code[[mutated_codon]]
              
              # Create the mutated sequence
              mutated_sequence <- full_sequence
              start <- (codon_index - 1) * 3 + 1
              substr(mutated_sequence, start, start+2) <- mutated_codon
              
              mutation_type <- if (mutated_aa == original_aa) "synonymous" else "missense"
              
              mutations <- rbind(mutations, data.frame(
                  codon_index = codon_index,
                  position = pos,
                  original_codon = codon,
                  mutated_codon = mutated_codon,
                  original_aa = original_aa,
                  mutated_aa = mutated_aa,
                  mutation_position = (codon_index -1)*3 + pos,
                  mutation_type = mutation_type,
                  sequence = mutated_sequence
              ))
          }
      }
  }
  return(mutations)
}

# Main function to process the whole sequence
mutate_sequence <- function(dna_sequence) {
  codons <- strsplit(dna_sequence, "")[[1]]
  codons <- sapply(seq(1, length(codons), by=3), function(i) paste(codons[i:(i+2)], collapse=""))
  all_mutations <- data.frame()
  
  for (i in seq_along(codons)) {
      codon <- codons[i]
      mutations <- mutate_codon(codon, i, dna_sequence)
      all_mutations <- rbind(all_mutations, mutations)
  }
  return(all_mutations)
}

# Example usage
sequence <- DRD2
mutations <- mutate_sequence(sequence)


# Filter synonymous and missense if needed
synonymous_mutations <- subset(mutations, mutation_type == "synonymous")
missense_mutations <- subset(mutations, mutation_type == "missense")

source <- c(NA,"wildtype",DRD2)

output <- rbind(source,mutations[,7:9])


write.csv(file="DRD2_mutations.csv",output)
