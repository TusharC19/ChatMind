"""New Code lines"""

import re
import pandas as pd


# ---------------------------------------------------------
# WhatsApp message timestamp pattern
#
# Supports:
# 12/09/26, 10:30 pm -
# 12/09/2026, 10:30 PM -
# ---------------------------------------------------------

MESSAGE_PATTERN = re.compile(
    r'(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s*'
    r'(?P<time>\d{1,2}:\d{2}\s*[AaPp][Mm])\s*-\s*'
)


def preprocess(data):
    """
    Convert exported WhatsApp chat text into a structured DataFrame.

    Parameters
    ----------
    data : str
        Raw WhatsApp exported chat text.

    Returns
    -------
    pandas.DataFrame
        Structured WhatsApp messages.
    """

    # ---------------------------------------------------------
    # 1. Validate input
    # ---------------------------------------------------------

    if not isinstance(data, str):
        raise TypeError("Input data must be a string.")

    if not data.strip():
        return pd.DataFrame(
            columns=[
                'message_id',
                'date',
                'user',
                'message',
                'year',
                'month_num',
                'month',
                'only_date',
                'day',
                'day_name',
                'hour',
                'minute',
                'period'
            ]
        )

    # ---------------------------------------------------------
    # 2. Find all WhatsApp message timestamps
    # ---------------------------------------------------------

    matches = list(MESSAGE_PATTERN.finditer(data))

    if not matches:
        raise ValueError(
            "No WhatsApp messages could be detected. "
            "Please check the exported WhatsApp chat format."
        )

    records = []

    # ---------------------------------------------------------
    # 3. Extract messages
    #
    # Everything between one timestamp and the next timestamp
    # belongs to the same message.
    #
    # This also handles multiline messages.
    # ---------------------------------------------------------

    for i, match in enumerate(matches):

        date_string = match.group('date')
        time_string = match.group('time')

        message_date = f"{date_string}, {time_string}"

        # Message begins after timestamp and " - "
        message_start = match.end()

        # Message ends before next timestamp
        if i + 1 < len(matches):
            message_end = matches[i + 1].start()
        else:
            message_end = len(data)

        raw_message = data[message_start:message_end].strip()

        # -----------------------------------------------------
        # 4. Separate user and message
        #
        # Normal message:
        #
        # Tushar: Hello bro
        #
        # Group/system notification:
        #
        # Messages and calls are end-to-end encrypted...
        # -----------------------------------------------------

        user_match = re.match(
            r'(?P<user>[^:\n]+):\s*(?P<message>.*)',
            raw_message,
            flags=re.DOTALL
        )

        if user_match:

            user = user_match.group('user').strip()
            message = user_match.group('message').strip()

        else:

            user = 'group_notification'
            message = raw_message.strip()

        records.append(
            {
                'date': message_date,
                'user': user,
                'message': message
            }
        )

    # ---------------------------------------------------------
    # 5. Create DataFrame
    # ---------------------------------------------------------

    df = pd.DataFrame(records)

    # ---------------------------------------------------------
    # 6. Convert date
    #
    # format='mixed' allows:
    # 12/09/26
    # 12/09/2026
    # ---------------------------------------------------------

    df['date'] = pd.to_datetime(
        df['date'],
        dayfirst=True,
        format='mixed',
        errors='coerce'
    )

    # Remove rows whose date couldn't be parsed
    df = df.dropna(subset=['date']).reset_index(drop=True)

    # ---------------------------------------------------------
    # 7. Clean user and message text
    # ---------------------------------------------------------

    df['user'] = (
        df['user']
        .astype(str)
        .str.strip()
    )

    df['message'] = (
        df['message']
        .astype(str)
        .str.strip()
    )

    # ---------------------------------------------------------
    # 8. Add unique message ID
    #
    # This will be useful later for:
    #
    # AI results
    # Semantic search
    # Source message tracking
    # Debugging
    # ---------------------------------------------------------

    df.insert(
        0,
        'message_id',
        range(1, len(df) + 1)
    )

    # ---------------------------------------------------------
    # 9. Date-related features
    # ---------------------------------------------------------

    df['year'] = df['date'].dt.year

    df['month_num'] = df['date'].dt.month

    df['month'] = df['date'].dt.month_name()

    df['only_date'] = df['date'].dt.date

    df['day'] = df['date'].dt.day

    df['day_name'] = df['date'].dt.day_name()

    df['hour'] = df['date'].dt.hour

    df['minute'] = df['date'].dt.minute

    # ---------------------------------------------------------
    # 10. Create hourly period
    #
    # Examples:
    #
    # 00 -> 00-01
    # 10 -> 10-11
    # 23 -> 23-00
    # ---------------------------------------------------------

    df['period'] = df['hour'].apply(
        lambda hour:
            f"{hour:02d}-00"
            if hour == 23
            else f"{hour:02d}-{hour + 1:02d}"
    )

    return df