resource "aws_lambda_function" "api" {
  count = var.deploy_function ? 1 : 0

  function_name                  = local.function
  role                           = aws_iam_role.lambda.arn
  package_type                   = "Image"
  image_uri                      = "${aws_ecr_repository.api.repository_url}:${var.image_tag}"
  architectures                  = ["x86_64"]
  memory_size                    = var.memory_mb
  timeout                        = var.timeout_seconds
  reserved_concurrent_executions = var.reserved_concurrency
  publish                        = true

  environment {
    variables = {
      SSM_PARAMETERS     = join(",", [for env, name in var.ssm_parameters : "${env}=${name}"])
      REQUIRE_API_KEY    = "true"
      API_KEY_STORE      = "ssm"
      API_KEYS_PARAMETER = var.api_keys_parameter
      QUERY_LOG_ENABLED  = "false"
    }
  }

  logging_config {
    log_format = "Text"
    log_group  = aws_cloudwatch_log_group.api.name
  }

  lifecycle {
    ignore_changes = [image_uri]

    precondition {
      condition     = var.image_tag != null
      error_message = "deploy_function = true necesita image_tag."
    }
  }
}

resource "aws_lambda_alias" "live" {
  count = var.deploy_function ? 1 : 0

  name             = "live"
  function_name    = aws_lambda_function.api[0].function_name
  function_version = aws_lambda_function.api[0].version

  lifecycle {
    ignore_changes = [function_version]
  }
}

resource "aws_lambda_function_url" "api" {
  count = var.deploy_function ? 1 : 0

  function_name      = aws_lambda_function.api[0].function_name
  qualifier          = aws_lambda_alias.live[0].name
  authorization_type = "NONE"
}

resource "aws_lambda_permission" "url" {
  count = var.deploy_function ? 1 : 0

  statement_id           = "url-publica"
  action                 = "lambda:InvokeFunctionUrl"
  function_name          = aws_lambda_function.api[0].function_name
  qualifier              = aws_lambda_alias.live[0].name
  principal              = "*"
  function_url_auth_type = "NONE"
}

resource "aws_lambda_permission" "url_invoke" {
  count = var.deploy_function ? 1 : 0

  statement_id             = "url-publica-invoke"
  action                   = "lambda:InvokeFunction"
  function_name            = aws_lambda_function.api[0].function_name
  qualifier                = aws_lambda_alias.live[0].name
  principal                = "*"
  invoked_via_function_url = true
}
