variable "region" {
  description = "Región de AWS. La organización de la cuenta solo permite us-east-2."
  type        = string
  default     = "us-east-2"
}

variable "profile" {
  description = "Perfil de la CLI de AWS con el usuario scribe-lab-deployer."
  type        = string
  default     = "scribe-lab"
}

variable "prefix" {
  description = "Prefijo de todos los recursos; la política del usuario solo permite este prefijo."
  type        = string
  default     = "scribe-lab"
}

variable "alert_email" {
  description = "Correo que recibe los avisos de Budgets y de las alarmas."
  type        = string
}

variable "budget_limits_usd" {
  description = "Montos mensuales que disparan un aviso de gasto."
  type        = list(number)
  default     = [1, 10]
}

variable "log_retention_days" {
  type    = number
  default = 7
}

variable "deploy_function" {
  description = "false crea solo la base (ECR, logs, rol, avisos); true crea además la Lambda y su URL."
  type        = bool
  default     = false
}

variable "image_tag" {
  description = "Tag de la imagen en ECR que usa la Lambda."
  type        = string
  default     = null
}

variable "reserved_concurrency" {
  description = "Instancias simultáneas máximas: el tope real contra abuso de la URL pública."
  type        = number
  default     = 5
}

variable "memory_mb" {
  type    = number
  default = 2048
}

variable "timeout_seconds" {
  type    = number
  default = 30
}

variable "ssm_parameters" {
  description = "Variables de entorno que la Lambda lee de SSM al arrancar."
  type        = map(string)
  default = {
    QDRANT_URL     = "/scribe-lab/qdrant_url"
    QDRANT_API_KEY = "/scribe-lab/qdrant_api_key"
  }
}

variable "api_keys_parameter" {
  type    = string
  default = "/scribe-lab/api_keys"
}
