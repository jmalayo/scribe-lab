resource "aws_cloudwatch_log_group" "api" {
  name              = "/aws/lambda/${local.function}"
  retention_in_days = var.log_retention_days
}
