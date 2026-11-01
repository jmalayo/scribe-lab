output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "function_url" {
  value = var.deploy_function ? aws_lambda_function_url.api[0].function_url : null
}

output "function_name" {
  value = local.function
}
