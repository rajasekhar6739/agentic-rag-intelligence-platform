class SelfCorrection:

    def __init__(
        self,
        max_retries=2
    ):
        self.max_retries = max_retries

    def should_retry(
        self,
        critic_result,
        attempt
    ):
        if attempt >= self.max_retries:
            return False

        if not critic_result:
            return True

        return not bool(
            critic_result.get(
                "supported",
                False
            )
        )

    def feedback(
        self,
        critic_result
    ):
        if not critic_result:
            return (
                "The previous answer could not be "
                "verified. Retrieve stronger evidence."
            )

        issues = critic_result.get(
            "issues",
            []
        )

        if not issues:
            return (
                "The previous answer was not sufficiently "
                "supported by the retrieved evidence."
            )

        return (
            "Improve the answer using stronger evidence. "
            "Issues identified: "
            + "; ".join(
                str(issue)
                for issue in issues
            )
        )