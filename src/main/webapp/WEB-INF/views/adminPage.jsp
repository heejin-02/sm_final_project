<%@ taglib uri="http://java.sun.com/jsp/jstl/core" prefix="c" %>
<%@ taglib uri="http://java.sun.com/jsp/jstl/fmt" prefix="fmt" %>
<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Insert title here</title>
</head>
<body>
<h1> 전체 회원 정보</h1>
<div>
	<button onclick="location.href='insertUser'" class="btn">회원 추가</button>
	<button onclick="location.href='insertFarm'" class="btn">(QC_Farm 테이블 정보)추가</button>
	<table>
		<tr>
			<th>번호</th>
			<th>이름</th>
			<th>아이디(휴대폰번호)</th>			
			<th>대표농장이름</th>
			<th>대표농장주소</th>
			<th>대표농장번호(회사번호)</th>
			<th>가입 날짜</th>
		</tr>
		<c:forEach var="farm" items="${farmList}">
		    <tr>
		        <td>${farm.farmIdx}</td>
				<td>
				  <a href="/admin/edit?userPhone=${farm.userPhone}"> ${farm.userName}</a>
				</td>

		        <td>${farm.userPhone}</td>
		        <td>${farm.farmName}</td>
		        <td>${farm.farmAddr}</td>
		        <td>${farm.farmPhone}</td>
		        <td>${farm.joinAt}</td>
		    </tr>
		</c:forEach>

	</table>
</div>
</body>
</html>