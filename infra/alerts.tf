resource "aws_budgets_budget" "monthly" {
  for_each = toset([for limit in var.budget_limits_usd : tostring(limit)])

  name         = "${var.prefix}-usd-${each.value}"
  budget_type  = "COST"
  limit_amount = each.value
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_types {
    include_credit = false
    include_refund = false
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.alert_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.alert_email]
  }
}

resource "aws_sns_topic" "alerts" {
  name = "${var.prefix}-alertas"
}

resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

resource "aws_cloudwatch_metric_alarm" "lambda" {
  for_each = var.deploy_function ? {
    errores   = "Errors"
    limitadas = "Throttles"
  } : {}

  alarm_name          = "${var.prefix}-api-${each.key}"
  alarm_description   = "La Lambda tuvo ${each.value} en los últimos 5 minutos."
  namespace           = "AWS/Lambda"
  metric_name         = each.value
  dimensions          = { FunctionName = local.function }
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  ok_actions          = [aws_sns_topic.alerts.arn]
}
