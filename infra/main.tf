terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# ── GCS bucket for dataset upload ──────────────────────────────────────────────

resource "google_storage_bucket" "data" {
  name                        = "${var.project_id}-audio-enhancer-data"
  location                    = var.region
  force_destroy               = true
  uniform_bucket_level_access = true

  lifecycle_rule {
    condition { age = 30 }
    action { type = "Delete" }
  }
}

# ── Firewall: allow SSH ───────────────────────────────────────────────────────

resource "google_compute_firewall" "allow_ssh" {
  name    = "audio-enhancer-allow-ssh"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["audio-enhancer"]
}

# ── Service account for the VM ────────────────────────────────────────────────

resource "google_service_account" "training_vm" {
  account_id   = "audio-enhancer-vm"
  display_name = "Audio Enhancer Training VM"
}

resource "google_project_iam_member" "vm_storage" {
  project = var.project_id
  role    = "roles/storage.objectUser"
  member  = "serviceAccount:${google_service_account.training_vm.email}"
}

# ── Persistent data disk (survives VM preemption / deletion) ──────────────────

resource "google_compute_disk" "data" {
  name = "audio-enhancer-data"
  type = "pd-standard" # HDD — cheapest (~$0.04/GB/month)
  size = var.data_disk_size_gb
  zone = var.zone
}

# ── GPU VM ────────────────────────────────────────────────────────────────────

resource "google_compute_instance" "training" {
  name         = "audio-enhancer-training"
  machine_type = var.machine_type # g2-standard-8 includes 1x L4 GPU
  zone         = var.zone
  tags         = ["audio-enhancer"]

  boot_disk {
    auto_delete = true
    initialize_params {
      image = "deeplearning-platform-release/pytorch-2-7-cu128-ubuntu-2404-nvidia-570"
      size  = var.boot_disk_size_gb
      type  = "pd-balanced"
    }
  }

  attached_disk {
    source      = google_compute_disk.data.self_link
    device_name = "audio-enhancer-data"
  }

  # No guest_accelerator block needed — G2 machine types include L4 GPUs

  scheduling {
    on_host_maintenance = "TERMINATE" # Required for GPU instances
    automatic_restart   = false
    provisioning_model  = "SPOT"      # Spot pricing (~70% cheaper)
  }

  network_interface {
    network = "default"
    access_config {} # Ephemeral public IP
  }

  service_account {
    email  = google_service_account.training_vm.email
    scopes = ["cloud-platform"]
  }

  metadata = {
    ssh-keys                = "${var.ssh_user}:${file(var.ssh_public_key_file)}"
    install-nvidia-driver   = "True"
    proxy-mode              = "project_editors"
  }

  metadata_startup_script = file("${path.module}/startup.sh")

  lifecycle {
    ignore_changes = [metadata_startup_script]
  }
}

# ── Outputs ───────────────────────────────────────────────────────────────────

output "vm_external_ip" {
  description = "Public IP of the training VM (use for VSCode Remote SSH)"
  value       = google_compute_instance.training.network_interface[0].access_config[0].nat_ip
}

output "vm_name" {
  description = "VM instance name"
  value       = google_compute_instance.training.name
}

output "gcs_bucket" {
  description = "GCS bucket for dataset upload"
  value       = google_storage_bucket.data.name
}

output "data_disk" {
  description = "Persistent data disk (survives VM deletion)"
  value       = "${google_compute_disk.data.name} (${google_compute_disk.data.size}GB pd-standard)"
}

output "ssh_command" {
  description = "SSH command to connect"
  value       = "ssh ${var.ssh_user}@${google_compute_instance.training.network_interface[0].access_config[0].nat_ip}"
}

output "vscode_remote_ssh" {
  description = "Add this to your ~/.ssh/config for VSCode Remote SSH"
  value       = <<-EOT
    Host audio-enhancer-gcp
      HostName ${google_compute_instance.training.network_interface[0].access_config[0].nat_ip}
      User ${var.ssh_user}
      IdentityFile ~/.ssh/id_ed25519
  EOT
}
