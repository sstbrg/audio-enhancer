variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone"
  type        = string
  default     = "us-central1-a"
}

variable "machine_type" {
  description = "VM machine type (G2 family includes L4 GPUs)"
  type        = string
  default     = "g2-standard-8" # 8 vCPU, 32GB RAM, 1x L4 24GB
}

variable "boot_disk_size_gb" {
  description = "Boot disk size in GB (DL VM image requires >= 100GB)"
  type        = number
  default     = 100
}

variable "data_disk_size_gb" {
  description = "Persistent data disk in GB (datasets + checkpoints, survives VM deletion)"
  type        = number
  default     = 300
}

variable "ssh_public_key_file" {
  description = "Path to SSH public key for VM access"
  type        = string
  default     = "~/.ssh/id_ed25519.pub"
}

variable "ssh_user" {
  description = "SSH username"
  type        = string
  default     = "stas"
}
